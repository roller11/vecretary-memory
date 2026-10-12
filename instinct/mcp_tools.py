# SPDX-License-Identifier: MPL-2.0
"""
Instinct - MCP Tools

MCP tools for VS Code Copilot integration, allowing Claude to
query and update the Instinct context store.
"""

import logging
from functools import wraps
from typing import Any, Dict, List, Optional

from .tool_contracts import MCPTool, MCPToolResult, ResultStatus, ToolParameter, ParamType

logger = logging.getLogger(__name__)


def _validated_execution(execute):
    @wraps(execute)
    async def bounded(tool, **kwargs):
        if tool.name == 'instinct_save_memory':
            from .write_policy import agent_write_error
            authority_error = agent_write_error(kwargs.get('importance_score', 70), kwargs.get('segment_type', 'context'))
            if authority_error:
                return MCPToolResult.error(authority_error)
        error = tool.validate_params(kwargs)
        if error:
            return MCPToolResult.error(error)
        return await execute(tool, **kwargs)
    return bounded

# Global store reference
_global_store = None


def _get_global_store():
    """Get the global Instinct store."""
    global _global_store
    return _global_store


def set_global_store(store) -> None:
    """Set the global Instinct store."""
    global _global_store
    _global_store = store


class InstinctGetContextTool(MCPTool):
    """Retrieve priority context from Instinct."""
    
    @_validated_execution
    async def execute(self, **kwargs) -> MCPToolResult:
        query = kwargs.get("query", "")
        max_tokens = kwargs.get("max_tokens", 8000)
        min_importance = kwargs.get("min_importance", 30)
        
        try:
            from instinct.retriever import get_retriever
            from instinct.models import RetrievalResult
            
            retriever = get_retriever()
            if not retriever:
                return MCPToolResult(
                    status=ResultStatus.ERROR,
                    error_message="Instinct not initialized.",
                )

            if kwargs.get("new_session", False):
                from .nudge import renew_mandatory
                renew_mandatory(retriever.store)
            
            result = retriever.retrieve(query=query, min_importance=min_importance, track_access=False)
            context_string = result.to_context_string(max_tokens=max_tokens)
            
            return MCPToolResult(
                status=ResultStatus.SUCCESS,
                data={
                    "context": context_string,
                    "validation_token": result.mandatory_verification["validation_token"],
                    "mandatory_count": len(result.mandatory),
                    "mandatory_titles": result.mandatory_titles,
                    "mandatory_fingerprint": result.mandatory_fingerprint,
                    "mandatory_verification": result.mandatory_verification,
                    "relevant_count": len(result.relevant),
                    "estimated_tokens": result.total_tokens_estimate,
                    "segments": [
                        {
                            "id": seg.id,
                            "title": seg.concept_title,
                            "score": seg.importance_score,
                            "memory_identifiers": seg.memory_identifiers,
                            "mandatory_scope": seg.mandatory_scope,
                            "mandatory_tags": seg.mandatory_tags,
                        }
                        for seg in (result.mandatory + result.relevant)
                    ],
                },
            )
        except Exception as e:
            logger.error("instinct_get_context failed; details withheld")
            return MCPToolResult.error("Context retrieval failed; details withheld")


class InstinctSaveMemoryTool(MCPTool):
    """Save important context to Instinct memory."""
    
    @_validated_execution
    async def execute(self, **kwargs) -> MCPToolResult:
        content = kwargs.get("content", "")
        concept_title = kwargs.get("concept_title", "")
        importance_score = kwargs.get("importance_score", 70)
        segment_type = kwargs.get("segment_type", "context")
        memory_identifiers = kwargs.get("memory_identifiers", [])
        mandatory_scope = kwargs.get("mandatory_scope", "always")
        mandatory_tags = kwargs.get("mandatory_tags", [])

        from .write_policy import agent_write_error
        authority_error = agent_write_error(importance_score, segment_type)
        if authority_error:
            return MCPToolResult.error(authority_error)
        
        try:
            from instinct.models import Segment, SegmentType, ScoredSegment, parse_memory_identifiers, parse_tag_list
            
            store = _get_global_store()
            if not store:
                return MCPToolResult(
                    status=ResultStatus.ERROR,
                    error_message="Instinct store not initialized",
                )
            
            seg_type = SegmentType(segment_type)
            segment = Segment(
                content=content,
                segment_type=seg_type,
                source="mcp_tool",
                metadata={
                    "mandatory_scope": mandatory_scope,
                    "mandatory_tags": parse_tag_list(mandatory_tags),
                },
            )
            scored = ScoredSegment(
                segment=segment,
                importance_score=importance_score,
                concept_title=concept_title,
                reasoning="Saved via MCP tool",
                memory_identifiers=parse_memory_identifiers(memory_identifiers),
            )
            segment_id = store.add(scored)
            
            return MCPToolResult(
                status=ResultStatus.SUCCESS,
                data={
                    "saved": True,
                    "segment_id": segment_id,
                    "importance_score": scored.importance_score,
                    "is_mandatory": scored.is_mandatory,
                    "memory_identifiers": scored.memory_identifiers,
                    "mandatory_scope": scored.mandatory_scope,
                    "mandatory_tags": scored.mandatory_tags,
                },
            )
        except Exception as e:
            logger.error("instinct_save_memory failed; details withheld")
            return MCPToolResult.error("Memory save failed; details withheld")


class InstinctGetMandatoryTool(MCPTool):
    """Get active mandatory items (score 99-100)."""
    
    @_validated_execution
    async def execute(self, **kwargs) -> MCPToolResult:
        try:
            from instinct.retriever import get_retriever
            from instinct.models import RetrievalResult
            
            retriever = get_retriever()
            if not retriever:
                return MCPToolResult(
                    status=ResultStatus.ERROR,
                    error_message="Instinct not initialized",
                )
            
            store = getattr(retriever, "store", None)
            if kwargs.get("new_session", False):
                if store is None:
                    return MCPToolResult.error("Mandatory renewal unavailable")
                from .nudge import renew_mandatory
                renew_mandatory(store)
            mandatory = retriever.retrieve_mandatory()
            access_batch = getattr(store, "record_access_batch", None)
            if mandatory and callable(access_batch) and not getattr(store, "_read_only", False):
                access_batch([segment.id for segment in mandatory])
            retrieval_result = RetrievalResult(mandatory=mandatory)
            items = [
                {
                    "title": seg.concept_title,
                    "content": seg.segment.content,
                    "type": seg.segment.segment_type.value,
                    "memory_identifiers": seg.memory_identifiers,
                    "mandatory_scope": seg.mandatory_scope,
                    "mandatory_tags": seg.mandatory_tags,
                }
                for seg in mandatory
            ]
            
            return MCPToolResult(
                status=ResultStatus.SUCCESS,
                data={
                    "mandatory_items": items,
                    "validation_token": retrieval_result.validation_token if mandatory else None,
                    "count": len(items),
                    "mandatory_titles": retrieval_result.mandatory_titles,
                    "mandatory_fingerprint": retrieval_result.mandatory_fingerprint,
                    "mandatory_verification": retrieval_result.mandatory_verification,
                },
            )
        except Exception as e:
            logger.error("instinct_get_mandatory failed; details withheld")
            return MCPToolResult.error("Mandatory retrieval failed; details withheld")


class InstinctSearchTool(MCPTool):
    """Search Instinct memory by query."""
    
    @_validated_execution
    async def execute(self, **kwargs) -> MCPToolResult:
        query = kwargs.get("query", "")
        limit = kwargs.get("limit", 10)
        min_score = kwargs.get("min_score", 0)
        
        try:
            from instinct.retriever import get_retriever

            retriever = get_retriever()
            if not retriever:
                return MCPToolResult(
                    status=ResultStatus.ERROR,
                    error_message="Instinct retriever not initialized",
                )
            
            retrieved = retriever.retrieve(
                query=query,
                max_items=limit,
                min_importance=min_score,
                track_access=False,
            )
            results = retrieved.relevant
            items = [
                {
                    "title": seg.concept_title,
                    "score": seg.importance_score,
                    "type": seg.segment.segment_type.value,
                    "content": seg.segment.content[:500],
                    "memory_identifiers": seg.memory_identifiers,
                    "mandatory_scope": seg.mandatory_scope,
                    "mandatory_tags": seg.mandatory_tags,
                }
                for seg in results
            ]
            
            return MCPToolResult(
                status=ResultStatus.SUCCESS,
                data={"results": items, "count": len(items), "query": query},
            )
        except Exception as e:
            logger.error("instinct_search failed; details withheld")
            return MCPToolResult.error("Memory search failed; details withheld")


class InstinctScoreDiagnosticsTool(MCPTool):
    """Get detailed scoring diagnostics for a segment."""
    
    @_validated_execution
    async def execute(self, **kwargs) -> MCPToolResult:
        segment_id = kwargs.get("segment_id", "")
        
        try:
            from instinct.retriever import get_retriever
            
            retriever = get_retriever()
            if not retriever:
                return MCPToolResult(
                    status=ResultStatus.ERROR,
                    error_message="Instinct not initialized",
                )
            
            diagnostics = retriever.get_score_diagnostics(segment_id)
            if not diagnostics:
                return MCPToolResult(
                    status=ResultStatus.ERROR,
                    error_message=f"Segment not found: {segment_id}",
                )
            
            return MCPToolResult(
                status=ResultStatus.SUCCESS,
                data=diagnostics,
            )
        except Exception as e:
            logger.error("instinct_score_diagnostics failed; details withheld")
            return MCPToolResult.error("Score diagnostics failed; details withheld")


class InstinctPipelineInfoTool(MCPTool):
    """Get information about the modifier pipeline configuration."""
    
    @_validated_execution
    async def execute(self, **kwargs) -> MCPToolResult:
        try:
            from instinct.retriever import get_retriever
            from instinct.decay import describe_decay_curve
            
            retriever = get_retriever()
            if not retriever:
                return MCPToolResult(
                    status=ResultStatus.ERROR,
                    error_message="Instinct not initialized",
                )
            
            pipeline = retriever.pipeline
            modifiers = [
                {
                    "name": m.name,
                    "priority": m.priority,
                    "enabled": m.enabled,
                }
                for m in pipeline.modifiers
            ]
            
            # Sample decay curves for reference
            decay_examples = [
                describe_decay_curve(score) for score in [100, 95, 80, 60, 40, 20, 10]
            ]
            
            return MCPToolResult(
                status=ResultStatus.SUCCESS,
                data={
                    "modifiers": modifiers,
                    "decay_curves": decay_examples,
                },
            )
        except Exception as e:
            logger.error("instinct_pipeline_info failed; details withheld")
            return MCPToolResult.error("Pipeline information failed; details withheld")


def get_instinct_tools() -> List[MCPTool]:
    """Get all Instinct MCP tools."""
    tools = [
        InstinctGetContextTool(
            name="instinct_get_context",
            description="Retrieve context from Instinct. Returns mandatory guardrails (score 99-100) and relevant context.",
            parameters=[
                ToolParameter(name="query", param_type=ParamType.STRING, description="Query to find context for", required=True),
                ToolParameter(name="max_tokens", param_type=ParamType.INTEGER, description="Max tokens", required=False, default=8000),
                ToolParameter(name="min_importance", param_type=ParamType.INTEGER, description="Min importance 0-100", required=False, default=30),
                ToolParameter(name="new_session", param_type=ParamType.BOOLEAN, description="True once at new chat/session or window reload to renew existing active mandatories", required=False, default=False),
            ],
        ),
        InstinctSaveMemoryTool(
            name="instinct_save_memory",
            description="Save durable nonmandatory context. Agent scores are capped at 92; mandatory records are owner-only.",
            parameters=[
                ToolParameter(name="content", param_type=ParamType.STRING, description="Content to store", required=True),
                ToolParameter(name="concept_title", param_type=ParamType.STRING, description="Brief title", required=True),
                ToolParameter(name="importance_score", param_type=ParamType.INTEGER, description="Agent score 0-92", required=False, default=70, min_value=0, max_value=92),
                ToolParameter(name="segment_type", param_type=ParamType.STRING, description="Nonmandatory type of content", required=False, default="context", enum=["architectural_decision", "implementation_detail", "context", "user_preference"]),
                ToolParameter(name="memory_identifiers", param_type=ParamType.STRING, description="Optional identifiers as 'term1||term2' or comma-separated text", required=False, default=""),
                ToolParameter(name="mandatory_scope", param_type=ParamType.STRING, description="Mandatory scope: always or conditional", required=False, default="always"),
                ToolParameter(name="mandatory_tags", param_type=ParamType.STRING, description="Applicability tags as 'tag1|tag2' or comma-separated text", required=False, default=""),
            ],
        ),
        InstinctGetMandatoryTool(
            name="instinct_get_mandatory",
            description="Get active mandatory (score 99-100) guardrails that MUST be followed.",
            parameters=[
                ToolParameter(name="new_session", param_type=ParamType.BOOLEAN, description="True once at new chat/session or window reload to renew existing active mandatories", required=False, default=False),
            ],
        ),
        InstinctSearchTool(
            name="instinct_search",
            description="Search Instinct memory for specific topics.",
            parameters=[
                ToolParameter(name="query", param_type=ParamType.STRING, description="Search query", required=True),
                ToolParameter(name="limit", param_type=ParamType.INTEGER, description="Max results", required=False, default=10),
                ToolParameter(name="min_score", param_type=ParamType.INTEGER, description="Min importance", required=False, default=0),
            ],
        ),
        InstinctScoreDiagnosticsTool(
            name="instinct_score_diagnostics",
            description="Get detailed scoring diagnostics for a segment, including decay, pop state, and audit trail.",
            parameters=[
                ToolParameter(name="segment_id", param_type=ParamType.STRING, description="Segment ID to diagnose", required=True),
            ],
        ),
        InstinctPipelineInfoTool(
            name="instinct_pipeline_info",
            description="Get information about the modifier pipeline configuration and decay curves.",
            parameters=[],
        ),
    ]
    text_limits = {'query': 8192, 'content': 65536, 'concept_title': 4096, 'segment_id': 160, 'segment_type': 64, 'memory_identifiers': 8192, 'mandatory_scope': 64, 'mandatory_tags': 8192}
    integer_limits = {'max_tokens': (1, 32000), 'limit': (1, 100), 'min_importance': (0, 100), 'min_score': (0, 100)}
    for tool in tools:
        for parameter in tool.parameters:
            if parameter.name in text_limits:
                parameter.max_length = text_limits[parameter.name]
            if parameter.name in integer_limits:
                parameter.min_value, parameter.max_value = integer_limits[parameter.name]
    return tools


def register_instinct_tools(registry) -> None:
    """Register Instinct tools with an MCP tool registry."""
    for tool in get_instinct_tools():
        registry.register(tool)
