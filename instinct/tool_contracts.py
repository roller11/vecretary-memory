# SPDX-License-Identifier: MPL-2.0
"""Neutral MCP tool contracts without legacy orchestration dependencies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional
import logging
import math

logger = logging.getLogger(__name__)


class ResultStatus(Enum):
    """Status of a tool execution."""

    SUCCESS = auto()
    ERROR = auto()
    PARTIAL = auto()                   


@dataclass
class MCPToolResult:
    """
    Result from executing a tool.

    This is what gets sent back to the AI model.
    """

    status: ResultStatus
    data: Any = None                          
    error_message: Optional[str] = None                           
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP-compatible dict."""
        result = {
            "status": self.status.name.lower(),
        }
        if self.data is not None:
            result["data"] = self.data
        if self.error_message:
            result["error"] = self.error_message
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def success(cls, data: Any, **metadata: Any) -> "MCPToolResult":
        """Create a success result."""
        return cls(status=ResultStatus.SUCCESS, data=data, metadata=metadata)

    @classmethod
    def error(cls, message: str, **metadata: Any) -> "MCPToolResult":
        """Create an error result."""
        return cls(status=ResultStatus.ERROR, error_message=message, metadata=metadata)


class ParamType(Enum):
    """Types of tool parameters."""

    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""

    name: str
    param_type: ParamType
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[str]] = None                  
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    max_length: Optional[int] = None

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format."""
        schema: Dict[str, Any] = {
            "type": self.param_type.value,
            "description": self.description,
        }
        if self.enum:
            schema["enum"] = self.enum
        if self.min_value is not None:
            schema["minimum"] = self.min_value
        if self.max_value is not None:
            schema["maximum"] = self.max_value
        if self.max_length is not None:
            schema["maxLength"] = self.max_length
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass
class MCPTool(ABC):
    """
    Base class for MCP tools.

    Subclass this to create new tools that AI agents can use.
    """

    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    category: str = "general"                      
    requires_confirmation: bool = False                        

    @abstractmethod
    async def execute(self, **kwargs: Any) -> MCPToolResult:
        """Execute the tool with given parameters."""
        pass

    def get_schema(self) -> Dict[str, Any]:
        """Get the tool schema in MCP format."""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        }

    def validate_params(self, params: Dict[str, Any]) -> Optional[str]:
        """
        Validate parameters before execution.

        Returns error message if invalid, None if valid.
        """
        if not isinstance(params, dict) or set(params) - {param.name for param in self.parameters}:
            return "Unknown tool parameters"
        for param in self.parameters:
            if param.required and param.name not in params:
                return f"Missing required parameter: {param.name}"

            if param.name in params:
                value = params[param.name]

                               
                if param.param_type == ParamType.STRING and not isinstance(value, str):
                    return f"Parameter '{param.name}' must be a string"
                if param.param_type == ParamType.NUMBER and (type(value) not in (int, float) or type(value) is float and not math.isfinite(value)):
                    return f"Parameter '{param.name}' must be a number"
                if param.param_type == ParamType.INTEGER and type(value) is not int:
                    return f"Parameter '{param.name}' must be an integer"
                if param.param_type == ParamType.STRING and param.max_length is not None and len(value) > param.max_length:
                    return f"Parameter '{param.name}' exceeds text limit"
                if param.param_type == ParamType.BOOLEAN and not isinstance(
                    value, bool
                ):
                    return f"Parameter '{param.name}' must be a boolean"

                               
                if param.enum and value not in param.enum:
                    return f"Parameter '{param.name}' must be one of: {param.enum}"

                                
                if param.min_value is not None and value < param.min_value:
                    return f"Parameter '{param.name}' must be at least {param.min_value}"
                if param.max_value is not None and value > param.max_value:
                    return f"Parameter '{param.name}' must be at most {param.max_value}"

        return None


class ToolRegistry:
    """Registry of all available MCP tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, MCPTool] = {}

    def register(self, tool: MCPTool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        logger.debug(f"Registered MCP tool: {tool.name}")

    def unregister(self, name: str) -> bool:
        """Unregister a tool."""
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get(self, name: str) -> Optional[MCPTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self, category: Optional[str] = None) -> List[MCPTool]:
        """List all registered tools."""
        tools = list(self._tools.values())
        if category:
            tools = [t for t in tools if t.category == category]
        return tools

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all tools (for MCP discovery)."""
        return [tool.get_schema() for tool in self._tools.values()]
