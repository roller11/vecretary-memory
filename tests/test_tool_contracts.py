# SPDX-License-Identifier: MPL-2.0
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from instinct.mcp_tools import get_instinct_tools, register_instinct_tools
from instinct.tool_contracts import (
    MCPTool,
    MCPToolResult,
    ParamType,
    ResultStatus,
    ToolParameter,
    ToolRegistry,
)


class ExampleTool(MCPTool):
    async def execute(self, **kwargs):
        return MCPToolResult.success(kwargs)


@pytest.mark.parametrize(
    ('result', 'expected'),
    [
        (MCPToolResult.success(False), {'status': 'success', 'data': False}),
        (MCPToolResult.success([], count=0), {'status': 'success', 'data': [], 'metadata': {'count': 0}}),
        (MCPToolResult.error('failure', retry=True), {'status': 'error', 'error': 'failure', 'metadata': {'retry': True}}),
        (MCPToolResult(ResultStatus.PARTIAL, data=0), {'status': 'partial', 'data': 0}),
        (MCPToolResult(ResultStatus.SUCCESS), {'status': 'success'}),
    ],
)
def test_result_serialization(result, expected):
    assert result.to_dict() == expected
    assert json.loads(json.dumps(result.to_dict())) == expected


def test_result_metadata_is_not_shared():
    first = MCPToolResult(ResultStatus.SUCCESS)
    second = MCPToolResult(ResultStatus.SUCCESS)
    first.metadata['test'] = True
    assert second.metadata == {}


@pytest.mark.parametrize('param_type', list(ParamType))
def test_parameter_schema_types(param_type):
    parameter = ToolParameter('value', param_type, 'A value')
    assert parameter.to_json_schema() == {'type': param_type.value, 'description': 'A value'}


def test_schema_preserves_defaults_enums_and_limits():
    tool = ExampleTool(
        'example',
        'Example',
        parameters=[
            ToolParameter('query', ParamType.STRING, 'Query', enum=['hello', 'world']),
            ToolParameter('limit', ParamType.NUMBER, 'Limit', required=False, default=0, min_value=0, max_value=10),
            ToolParameter('enabled', ParamType.BOOLEAN, 'Enabled', required=False, default=False),
        ],
    )
    schema = tool.get_schema()
    assert schema['name'] == 'example'
    assert schema['description'] == 'Example'
    assert schema['inputSchema']['required'] == ['query']
    assert schema['inputSchema']['properties']['query']['enum'] == ['hello', 'world']
    assert schema['inputSchema']['properties']['limit'] == {
        'type': 'number', 'description': 'Limit', 'minimum': 0, 'maximum': 10, 'default': 0,
    }
    assert schema['inputSchema']['properties']['enabled']['default'] is False
    assert tool.validate_params({'query': 'hello'}) is None


@pytest.mark.parametrize(
    ('parameter', 'params', 'expected'),
    [
        (ToolParameter('value', ParamType.STRING, 'Value'), {}, 'Missing required parameter: value'),
        (ToolParameter('value', ParamType.STRING, 'Value'), {'value': 1}, "Parameter 'value' must be a string"),
        (ToolParameter('value', ParamType.NUMBER, 'Value'), {'value': '1'}, "Parameter 'value' must be a number"),
        (ToolParameter('value', ParamType.BOOLEAN, 'Value'), {'value': 1}, "Parameter 'value' must be a boolean"),
        (ToolParameter('value', ParamType.STRING, 'Value', enum=['yes']), {'value': 'no'}, "Parameter 'value' must be one of: ['yes']"),
        (ToolParameter('value', ParamType.NUMBER, 'Value', min_value=0), {'value': -1}, "Parameter 'value' must be at least 0"),
        (ToolParameter('value', ParamType.NUMBER, 'Value', max_value=10), {'value': 11}, "Parameter 'value' must be at most 10"),
        (ToolParameter('value', ParamType.NUMBER, 'Value', min_value=0, max_value=10), {'value': 0}, None),
        (ToolParameter('value', ParamType.NUMBER, 'Value', min_value=0, max_value=10), {'value': 10}, None),
    ],
)
def test_supported_parameter_validation(parameter, params, expected):
    tool = ExampleTool('example', 'Example', parameters=[parameter])
    assert tool.validate_params(params) == expected


def test_tool_base_is_abstract():
    with pytest.raises(TypeError):
        MCPTool('abstract', 'Abstract')


def test_registry_registration_filtering_replacement_and_removal():
    registry = ToolRegistry()
    first = ExampleTool('first', 'First', category='memory')
    second = ExampleTool('second', 'Second', category='other')
    registry.register(first)
    registry.register(second)
    assert registry.get('first') is first
    assert registry.get('missing') is None
    assert registry.list_tools() == [first, second]
    assert registry.list_tools('memory') == [first]
    assert registry.list_tools('missing') == []
    assert registry.get_schemas() == [first.get_schema(), second.get_schema()]
    replacement = ExampleTool('first', 'Replacement')
    registry.register(replacement)
    assert registry.get('first') is replacement
    assert len(registry.list_tools()) == 2
    assert registry.unregister('first') is True
    assert registry.unregister('first') is False
    assert registry.list_tools() == [second]


def test_instinct_tools_use_neutral_contracts():
    expected_names = {
        'instinct_get_context', 'instinct_save_memory', 'instinct_get_mandatory',
        'instinct_search', 'instinct_score_diagnostics', 'instinct_pipeline_info',
    }
    tools = get_instinct_tools()
    assert {tool.name for tool in tools} == expected_names
    assert all(isinstance(tool, MCPTool) for tool in tools)
    registry = ToolRegistry()
    register_instinct_tools(registry)
    assert {schema['name'] for schema in registry.get_schemas()} == expected_names
    assert registry.get('instinct_get_context').get_schema()['inputSchema']['properties']['max_tokens']['default'] == 8000
    assert registry.get('instinct_save_memory').validate_params({}) == 'Missing required parameter: content'


def test_fresh_process_discovery_without_legacy_or_premium_imports(tmp_path):
    workspace = Path(__file__).resolve().parents[1]
    script = '''
import importlib.abc
import json
from pathlib import Path
import sys

denied = ('tc_mcp', 'core', 'providers', 'wisdom', 'triage', 'instinct.cot')

class DenyLegacyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == prefix or fullname.startswith(prefix + '.') for prefix in denied):
            raise AssertionError('Unexpected legacy/premium import: ' + fullname)

sys.meta_path.insert(0, DenyLegacyImports())
import instinct
from instinct.tool_contracts import ToolRegistry

assert Path(instinct.__file__).resolve() == Path.cwd() / 'instinct' / '__init__.py'
assert 'instinct.mcp_tools' not in sys.modules
discover, register = instinct.get_mcp_tools()
registry = ToolRegistry()
register(registry)
assert len(discover()) == len(registry.get_schemas()) == 6
assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules for prefix in denied)
print(json.dumps({'module': instinct.__file__, 'tools': len(registry.list_tools())}))
'''
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)
    environment['CE_DATA_DIR'] = str(tmp_path / 'data')
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    completed = subprocess.run(
        [sys.executable, '-c', script], cwd=workspace, env=environment,
        text=True, capture_output=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout)
    assert Path(result['module']).resolve() == workspace / 'instinct' / '__init__.py'
    assert result['tools'] == 6
