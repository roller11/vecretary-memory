# SPDX-License-Identifier: MPL-2.0
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    'chroma_candidate_builder', Path(__file__).resolve().parents[1] / 'tools/build_chroma_security_candidate.py',
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_exact_patch_context_can_be_relocated_without_fuzzy_matching():
    patch = '@@ -1,2 +1,3 @@\n context\n+inserted\n ending'
    assert builder.apply_hunks('prefix\ncontext\nending\n', patch) == 'prefix\ncontext\ninserted\nending\n'


@pytest.mark.parametrize('original', ['different\n', 'context\nending\ncontext\nending\n'])
def test_missing_or_ambiguous_context_fails_closed(original):
    with pytest.raises(ValueError, match='context'):
        builder.apply_hunks(original, '@@ -1,2 +1,3 @@\n context\n+inserted\n ending')


def test_new_upstream_module_requires_an_empty_original():
    patch = '@@ -0,0 +1,1 @@\n+safe'
    assert builder.apply_hunks('', patch) == 'safe\n'
    with pytest.raises(ValueError):
        builder.apply_hunks('existing\n', patch)


def test_unreviewed_base_wheel_is_refused_before_network_or_output(tmp_path):
    base = tmp_path / 'synthetic.whl'
    base.write_bytes(b'synthetic unreviewed bytes')
    target = tmp_path / 'must-not-create'
    with pytest.raises(ValueError, match='Unreviewed'):
        builder.build(base, target)
    assert not target.exists()


def test_http_server_replacements_are_traceable_and_leave_binary_unchanged():
    binding_source = b'from .chromadb_rust_bindings import *\n'
    contents = {name: b'synthetic server source\n' for name in builder.SERVER_PATHS}
    contents['chromadb_rust_bindings/__init__.py'] = binding_source
    contents['chromadb_rust_bindings/native.so'] = b'synthetic native bytes'
    changes = builder.disable_http_servers(contents)
    assert {change['path'] for change in changes} == builder.SERVER_PATHS
    assert contents['chromadb_rust_bindings/native.so'] == b'synthetic native bytes'
    for name in ('chromadb/server/__init__.py', 'chromadb/server/fastapi/__init__.py'):
        with pytest.raises(RuntimeError, match='HTTP servers.*disabled'):
            exec(contents[name], {})
    namespace = {}
    exec(contents['chromadb/cli/cli.py'], namespace)
    with pytest.raises(RuntimeError, match='HTTP servers.*disabled'):
        namespace['app']()
    assert b'chromadb_rust_bindings.cli = cli' in contents['chromadb_rust_bindings/__init__.py']


def test_missing_or_unexpected_server_sources_are_refused():
    with pytest.raises(ValueError, match='Missing reviewed'):
        builder.disable_http_servers({})
    with pytest.raises(ValueError, match='Unexpected native'):
        builder.disable_http_servers({name: b'unreviewed' for name in builder.SERVER_PATHS})
