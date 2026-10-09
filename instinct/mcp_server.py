# SPDX-License-Identifier: MPL-2.0
"""Vecretary Memory stdio MCP server with existing memory write guards."""


import os
os.environ["TQDM_DISABLE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
import sys
import json
import math
import time
from contextvars import ContextVar
from .privacy_logging import MCPPrivacyFormatter
from .service_binding import bound_store, run_store_call


_stderr_handler = logging.StreamHandler(sys.stderr)
_stderr_handler.setFormatter(MCPPrivacyFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[_stderr_handler])
logger = logging.getLogger("instinct-mcp")


try:
    from mcp.server import Server
    from mcp import types
except ImportError as e:
    logger.error("MCP SDK not installed")
    logger.error("Install with: pip install mcp")
    sys.exit(1)













def _probe_embedding_readiness() -> dict:
    return {"ready": False, "error": "Embedding readiness is managed by the explicit SQLite service configuration."}















from .write_policy import MAX_AGENT_SCORE

MCP_SCORE_CAP = MAX_AGENT_SCORE


_highpri_write_timestamps: list[float] = []
HIGHPRI_WRITE_LIMIT = 5
HIGHPRI_WRITE_WINDOW = 300
MIN_HIGHPRI_CONTENT_LEN = 50








SEARCH_SIMILARITY_FLOOR = 0.60


BLOCKED_CONTENT_PATTERNS = [
    "call get_mandatory",
    "no more save_memory",
    "cease save_memory",
    "stop calling save_memory",
    "stop recursion",
]

BLOCKED_TITLE_PATTERNS = [
    "stop", "cease", "halt", "final", "break loop",
    "immediate halt", "stop now", "stop immediately",
]


def _check_write_guards(content: str, title: str, importance: int, *, track_rate=True) -> str | None:
    """
    Validate a save_memory request. Returns error message if blocked, None if OK.
    """
    content_lower = content.lower().strip()
    title_lower = title.lower().strip()


    if importance > MCP_SCORE_CAP:
        return (
            f"Blocked: MCP tool is capped at score {MCP_SCORE_CAP}. "
            f"Scores {MCP_SCORE_CAP + 1}-100 are owner-only (set via admin tools or scripts). "
            f"Requested score: {importance}"
        )


    for pattern in BLOCKED_CONTENT_PATTERNS:
        if pattern in content_lower and len(content) < 200:
            return f"Blocked: content matches noise pattern '{pattern}'"


    if title_lower in BLOCKED_TITLE_PATTERNS:
        return f"Blocked: title '{title}' matches noise pattern"


    if importance >= 80:

        if len(content.strip()) < MIN_HIGHPRI_CONTENT_LEN:
            return (
                f"Blocked: score {importance} requires at least "
                f"{MIN_HIGHPRI_CONTENT_LEN} chars of content "
                f"(got {len(content.strip())})"
            )


        if not track_rate:
            return None
        now = time.time()
        _highpri_write_timestamps[:] = [
            ts for ts in _highpri_write_timestamps
            if now - ts < HIGHPRI_WRITE_WINDOW
        ]
        if len(_highpri_write_timestamps) >= HIGHPRI_WRITE_LIMIT:
            return (
                f"Blocked: rate limit reached ({HIGHPRI_WRITE_LIMIT} "
                f"score-80+ writes per {HIGHPRI_WRITE_WINDOW}s window). "
                f"Wait before saving more high-priority items."
            )
        _highpri_write_timestamps.append(now)

    return None


def _check_dedup(store, content: str, title: str) -> str | None:
    """
    Check for near-duplicate entries. Returns warning message if dupe found.
    """

    existing = store.search(query=title, limit=3, min_importance=0)
    for seg in existing:
        existing_title = seg.concept_title.lower().strip()
        new_title = title.lower().strip()

        if existing_title == new_title:
            return (
                f"Duplicate detected: entry with title '{seg.concept_title}' "
                f"already exists (ID: {seg.id}, score: {seg.importance_score}). "
                f"Use instinct_search to find and review it first."
            )
    return None






_backend_lifecycle = ContextVar('instinct_backend_lifecycle', default=None)


server = Server("vecretary-sqlite-tools")


def _connection_lifecycle():
    backend = _backend_lifecycle.get()
    if backend is not None:
        return backend
    try:
        return server.request_context.lifespan_context.get("mandatory_lifecycle")
    except LookupError:
        return None


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """Return list of available instinct tools."""
    tools = [
        types.Tool(
            name="instinct_get_context",
            description="Retrieve relevant context for a query. Returns priority-scored context segments including mandatory standing instructions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to find relevant context for"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10
                    },
                    "include_mandatory": {
                        "type": "boolean",
                        "description": "Include all mandatory (score=100) items",
                        "default": True
                    },
                    "new_session": {
                        "type": "boolean",
                        "description": "True once on a new chat/session or window reload: renew existing active mandatory memories to 100 before retrieval.",
                        "default": False
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="instinct_save_memory",
            description="Store a new memory segment with importance scoring. Use for important decisions, patterns, or user preferences.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content to store"
                    },
                    "title": {
                        "type": "string",
                        "description": "Brief title for the concept"
                    },
                    "importance": {
                        "type": "integer",
                        "description": "Agent importance score 0-92; higher scores are owner-only",
                        "minimum": 0,
                        "maximum": MCP_SCORE_CAP,
                        "default": 70
                    },
                    "segment_type": {
                        "type": "string",
                        "description": "Type of segment",
                        "enum": [
                            "architectural_decision",
                            "implementation_detail",
                            "context",
                            "user_preference"
                        ],
                        "default": "context"
                    }
                },
                "required": ["content", "title"]
            }
        ),
        types.Tool(
            name="instinct_get_mandatory",
            description="Get all mandatory (score=100) context items. These are non-negotiable constraints and instructions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "new_session": {
                        "type": "boolean",
                        "description": "True once on a new chat/session or window reload to renew existing active mandatories.",
                        "default": False
                    }
                },
                "required": []
            }
        ),
        types.Tool(
            name="instinct_search",
            description="Semantic search across stored context with score filtering.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "min_score": {
                        "type": "integer",
                        "description": "Minimum importance score filter",
                        "default": 0
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results",
                        "default": 10
                    },
                    "min_similarity": {
                        "type": "number",
                        "description": "Cosine similarity floor 0-1 (default 0.60; 0 disables). Filters noise on vague queries.",
                        "default": 0.60
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="instinct_get",
            description="Fetch a single memory segment by its exact ID (full, untruncated content). Use when a search result was truncated or you know the ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "segment_id": {
                        "type": "string",
                        "description": "The exact segment ID, e.g. 20260930143504_2cc451ef3cb2b81a"
                    }
                },
                "required": ["segment_id"]
            }
        ),
        types.Tool(
            name="instinct_delete_memory",
            description="Move a memory to the recoverable recycle bin. This does not permanently delete it; restore it from the dashboard recycle bin. Owner-only records cannot be recycled by agent tools.",
            inputSchema={
                "type": "object",
                "properties": {
                    "segment_id": {
                        "type": "string",
                        "description": "The exact ID of the memory to move to the recycle bin"
                    }
                },
                "required": ["segment_id"]
            }
        ),
        types.Tool(
            name="instinct_update_score",
            description="Update the importance score of an existing memory segment. Use instinct_search first to find the ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "segment_id": {
                        "type": "string",
                        "description": "The ID of the segment to update"
                    },
                    "new_score": {
                        "type": "integer",
                        "description": "New importance score (0-100)"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why the score is being changed",
                        "default": ""
                    }
                },
                "required": ["segment_id", "new_score"]
            }
        )
    ]
    text_limits = {'query': 8192, 'content': 65536, 'title': 4096, 'reason': 4096, 'segment_id': 160, 'segment_type': 64}
    integer_limits = {'max_results': (1, 100), 'limit': (1, 100), 'min_score': (0, 100), 'importance': (0, MCP_SCORE_CAP), 'new_score': (0, MCP_SCORE_CAP)}
    for tool in tools:
        tool.inputSchema['additionalProperties'] = False
        for field, schema in tool.inputSchema['properties'].items():
            if field in text_limits:
                schema['maxLength'] = text_limits[field]
            if field in integer_limits:
                schema['minimum'], schema['maximum'] = integer_limits[field]
            if field == 'min_similarity':
                schema.update(minimum=0, maximum=1)
    return tools


def _validate_tool_bounds(arguments):
    if not isinstance(arguments, dict) or len(arguments) > 16:
        raise ValueError('Invalid tool object')
    for name, maximum in [('query', 8192), ('content', 65536), ('title', 4096), ('reason', 4096), ('segment_id', 160), ('segment_type', 64)]:
        if name in arguments and (not isinstance(arguments[name], str) or len(arguments[name]) > maximum):
            raise ValueError('Invalid tool text')
    for name, minimum, maximum in [('max_results', 1, 100), ('limit', 1, 100), ('max_tokens', 1, 32000), ('min_score', 0, 100), ('importance', 0, 100), ('new_score', 0, 100)]:
        if name in arguments and (type(arguments[name]) is not int or not minimum <= arguments[name] <= maximum):
            raise ValueError('Invalid tool integer')
    for flag in ('include_mandatory', 'new_session'):
        if flag in arguments and type(arguments[flag]) is not bool:
            raise ValueError('Invalid tool flag')
    if 'min_similarity' in arguments:
        value = arguments['min_similarity']
        if type(value) not in (int, float) or not 0 <= value <= 1 or not math.isfinite(value):
            raise ValueError('Invalid similarity floor')


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Execute a tool and return results."""
    known_tools = {
        "instinct_get_context", "instinct_save_memory", "instinct_get_mandatory",
        "instinct_search", "instinct_get", "instinct_delete_memory", "instinct_update_score",
    }
    log_name = name if name in known_tools else "unknown"
    logger.info("Tool called: %s", log_name)
    try:
        if name not in known_tools:
            raise ValueError('Unknown tool')
        _validate_tool_bounds(arguments)
        definitions = await list_tools()
        schema = next(tool.inputSchema for tool in definitions if tool.name == name)
        if set(arguments) - set(schema['properties']) or set(schema['required']) - set(arguments):
            raise ValueError('Unknown or missing tool fields')
    except ValueError:
        return [types.TextContent(type='text', text='Error: invalid tool arguments or unknown tool; nothing executed.')]

    try:
        store = bound_store()
    except Exception:
        logger.error("SQLite service binding unavailable for tool call: %s", log_name)
        return [types.TextContent(
            type="text",
            text="Error: SQLite service binding unavailable; nothing was read or written.",
        )]
    if store is None:
        logger.error("SQLite service binding missing for tool call: %s", log_name)
        return [types.TextContent(
            type="text",
            text="Error: No SQLite service is bound; nothing was read or written. Start the explicit vecretary-mcp service."
        )]

    if store._read_only and name in ('instinct_save_memory', 'instinct_update_score', 'instinct_delete_memory'):
        if getattr(store, 'queue_writes', False):
            try:
                from .sqlite_pending import capture_call
                return [types.TextContent(type='text', text=capture_call(store, name, arguments))]
            except Exception:
                return [types.TextContent(type='text', text='Error: SQLite fallback capture unconfirmed; inspect the private pending queue and committed records before retrying. The observer did not write the database; queued replay may still occur.')]
        return [types.TextContent(type='text', text='Error: SQLite observer is read-only; no queued write was created.')]

    try:

        try:
            from instinct.nudge import tick_nudge_session
            from instinct.dormancy import DormancyConfig, _cfg as _dorm_cfg
            idle_min = float(_dorm_cfg().session_idle_minutes)
            if not store._read_only:
                tick_nudge_session(store, idle_minutes=idle_min)
        except Exception:
            logger.error("Session tick failed (non-fatal)")
            raise




        try:
            from instinct import tick_turn, apply_nudge_check
            if not store._read_only:
                tick_turn(store)
        except Exception:
            logger.error("Nudge tick failed (non-fatal)")
            raise






        if name in ("instinct_get_context", "instinct_get_mandatory") and not store._read_only:
            try:
                from instinct import apply_nudge_check
                apply_nudge_check(store)
            except Exception:
                logger.error("Nudge check failed (non-fatal)")
                raise









        lifecycle = _connection_lifecycle()
        if arguments.get("new_session", False):
            if lifecycle is None:
                from .mandatory_lifecycle import MandatoryLifecycle
                lifecycle = MandatoryLifecycle()
            else:
                lifecycle.request()
        if lifecycle is not None:
            renewed = await run_store_call(store, lifecycle.restore_if_pending, store)
            if not renewed and name in ("instinct_get_context", "instinct_get_mandatory"):
                return [types.TextContent(
                    type="text",
                    text="Error: Mandatory renewal is pending until an owner SQLite service can write it.",
                )]

        if name == "instinct_get_context":
            return await _handle_get_context(store, arguments)
        elif name == "instinct_save_memory":
            return await _handle_save_memory(store, arguments)
        elif name == "instinct_get_mandatory":
            return await _handle_get_mandatory(store, arguments)
        elif name == "instinct_search":
            return await _handle_search(store, arguments)
        elif name == "instinct_get":
            return await _handle_get(store, arguments)
        elif name == "instinct_delete_memory":
            return await _handle_delete_memory(store, arguments)
        elif name == "instinct_update_score":
            return await _handle_update_score(store, arguments)
        else:
            return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        logger.error("Tool execution failed: %s", log_name)
        return [types.TextContent(
            type="text",
            text=f"Error executing {log_name}; exception details withheld. Check provider/settings and retry. Server logs contain metadata only."
        )]






async def _handle_get_context(store, arguments: dict) -> list[types.TextContent]:
    """Handle instinct_get_context tool using PriorityRetriever.

    Uses the full modifier pipeline (decay, pop mechanics) so results
    are ranked by effective_score rather than raw cosine distance alone.
    """
    from instinct.retriever import PriorityRetriever, RetrievalConfig

    query = arguments.get("query", "")
    max_results = arguments.get("max_results", 10)
    max_tokens = arguments.get("max_tokens", 8000)
    include_mandatory = arguments.get("include_mandatory", True)

    config = RetrievalConfig(
        always_include_mandatory=include_mandatory,
        max_relevant_items=max_results,
        track_access=True,
    )
    retriever = PriorityRetriever(store=store, config=config)


    result = await run_store_call(store, retriever.retrieve, query)

    total = len(result.mandatory) + len(result.relevant)
    context_text = result.to_context_string(max_tokens=max_tokens)



    final_text = f"Found {total} context items:\n\n{context_text}" if total else "No context found."

    return [types.TextContent(
        type="text",
        text=final_text
    )]


async def _handle_save_memory(store, arguments: dict) -> list[types.TextContent]:
    """Handle instinct_save_memory tool with write guards."""
    from instinct.models import Segment, ScoredSegment, SegmentType
    from datetime import datetime

    content = arguments.get("content", "")
    title = arguments.get("title", "Untitled")



    importance = arguments.get("importance", 70)
    seg_type_str = arguments.get("segment_type", "context")

    from .write_policy import agent_write_error
    authority_error = agent_write_error(importance, seg_type_str)
    if authority_error:
        return [types.TextContent(type="text", text=f"BLOCKED: {authority_error}")]


    guard_error = _check_write_guards(content, title, importance)
    if guard_error:
        logger.warning("Write guard blocked save_memory")
        return [types.TextContent(
            type="text",
            text=f"BLOCKED: {guard_error}"
        )]

    dedup_warning = _check_dedup(store, content, title)
    if dedup_warning:
        logger.warning("Dedup check flagged save_memory")
        return [types.TextContent(
            type="text",
            text=f"WARNING: {dedup_warning}"
        )]



    type_map = {
        "standing_instruction": SegmentType.STANDING_INSTRUCTION,
        "architectural_decision": SegmentType.ARCHITECTURAL_DECISION,
        "critical_constraint": SegmentType.CRITICAL_CONSTRAINT,
        "implementation_detail": SegmentType.IMPLEMENTATION_DETAIL,
        "context": SegmentType.CONTEXT,
        "user_preference": SegmentType.USER_PREFERENCE,
    }
    seg_type = type_map.get(seg_type_str, SegmentType.CONTEXT)

    segment = Segment(
        content=content,
        segment_type=seg_type,
        timestamp=datetime.now(),
        source="mcp",
    )
    scored = ScoredSegment(
        segment=segment,
        importance_score=importance,
        concept_title=title,
        reasoning="Saved via MCP tool",
    )








    segment_id = store.add(scored)



    return [types.TextContent(
        type="text",
        text=f"✓ Saved memory segment\nID: {segment_id}\nTitle: {title}\nImportance: {importance}"
    )]


async def _handle_get_mandatory(store, arguments: dict) -> list[types.TextContent]:
    """Handle instinct_get_mandatory tool."""
    from instinct.models import RetrievalResult

    mandatory = store.get_mandatory()
    retrieval_result = RetrievalResult(mandatory=mandatory)
    content = (
        f"Found {len(mandatory)} mandatory items:\n\n{retrieval_result.to_context_string()}"
        if mandatory else
        "No mandatory items found."
    )

    return [types.TextContent(
        type="text",
        text=content
    )]


def _mandatory_acknowledgement(segments) -> str:
    """Challenge explicit mandatory reads without renewing or accessing the store."""
    from instinct.models import RetrievalResult

    return RetrievalResult(relevant=segments).acknowledgement_context()


async def _handle_search(store, arguments: dict) -> list[types.TextContent]:
    """Handle instinct_search tool with effective score visibility."""
    from instinct.modifiers import ModifierPipeline, QueryContext, create_default_pipeline

    query = arguments.get("query", "")
    min_score = arguments.get("min_score", 0)
    limit = arguments.get("limit", 10)










    floor = arguments.get("min_similarity", SEARCH_SIMILARITY_FLOOR)


    segments = await run_store_call(
        store,
        lambda: store.search(
            query=query,
            limit=limit,
            min_importance=min_score if min_score > 0 else None,
            min_similarity=max(0.0, float(floor)),
        )
    )


    pipeline = create_default_pipeline()
    context = QueryContext.from_query(query)
    context.persist_dir = store.persist_dir








    def _clip(text: str, max_chars: int = 1200) -> str:
        if len(text) <= max_chars:
            return text
        cut = text[:max_chars]

        nl = cut.rfind("\n")
        if nl > max_chars // 2:
            cut = cut[:nl]
        return f"{cut}\n[... truncated ({len(text) - len(cut)} chars) — use instinct_get(id) for full content]"

    results = []
    for s in segments:
        scoring = pipeline.apply(s, context)
        if scoring.effective_score != s.importance_score:
            score_label = f"{scoring.effective_score:.0f}←{s.importance_score}"
        else:
            score_label = str(s.importance_score)
        results.append(
            f"[{score_label}] {s.concept_title}\nID: {s.id}\n{_clip(s.segment.content)}"
        )

    acknowledgement = _mandatory_acknowledgement(segments)
    response = f"Found {len(segments)} matching items:\n\n" + "\n\n---\n\n".join(results) if results else "No results found."
    if acknowledgement:
        response = acknowledgement + "\n\n" + response

    return [types.TextContent(
        type="text",
        text=response
    )]


async def _handle_get(store, arguments: dict) -> list[types.TextContent]:
    """Handle instinct_get tool: fetch one segment by exact ID, untruncated.

    GET_BY_ID_TOOL (removable): pairs with SEARCH_INLINE_TRUNCATE — search
    results clip oversized content and point here for the full body.
    To remove: delete this handler, its dispatch line, and the Tool
    registration block.
    """
    segment_id = arguments.get("segment_id", "")
    if not segment_id:
        return [types.TextContent(type="text", text="Error: segment_id is required")]

    segment = store.get(segment_id)
    if not segment:
        return [types.TextContent(
            type="text",
            text=f"Error: No segment found with ID '{segment_id}'"
        )]

    text = (
        f"[{segment.importance_score}] {segment.concept_title}\n"
        f"ID: {segment.id}\n"
        f"Type: {segment.segment.segment_type.value} | "
        f"Created: {segment.segment.timestamp.isoformat()}\n"
        f"---\n"
        f"{segment.segment.content}"
    )
    acknowledgement = _mandatory_acknowledgement([segment])
    if acknowledgement:
        text = acknowledgement + "\n\n" + text
    return [types.TextContent(type="text", text=text)]


async def _handle_update_score(store, arguments: dict) -> list[types.TextContent
]:
    """Handle instinct_update_score tool."""
    segment_id = arguments.get("segment_id", "")
    new_score = arguments.get("new_score", -1)
    reason = arguments.get("reason", "Score updated via MCP tool")

    if not segment_id:
        return [types.TextContent(type="text", text="Error: segment_id is required")]
    if type(new_score) is not int or not 0 <= new_score <= 100:
        return [types.TextContent(type="text", text="Error: new_score must be 0-100")]
    if new_score > MCP_SCORE_CAP:
        return [types.TextContent(
            type="text",
            text=f"Blocked: MCP tool is capped at score {MCP_SCORE_CAP}. "
                 f"Scores {MCP_SCORE_CAP + 1}-100 are owner-only."
        )]


    segment = store.get(segment_id)
    if not segment:
        return [types.TextContent(
            type="text",
            text=f"Error: No segment found with ID '{segment_id}'"
        )]

    from .write_policy import agent_target_error
    authority_error = agent_target_error(segment)
    if authority_error:
        return [types.TextContent(type="text", text=f"BLOCKED: {authority_error}")]
    old_score = segment.importance_score
    title = segment.concept_title
    success = store.update_score(segment_id, new_score, reason)

    if success:
        logger.info("Segment score updated")
        return [types.TextContent(
            type="text",
            text=f"Updated score\nID: {segment_id}\nTitle: {title}\nScore: {old_score} -> {new_score}\nReason: {reason}"
        )]
    else:
        return [types.TextContent(
            type="text",
            text=f"Error: Failed to update score for '{segment_id}'"
        )]






async def _handle_delete_memory(store, arguments: dict) -> list[types.TextContent]:
    segment_id = arguments.get('segment_id', '')
    if not isinstance(segment_id, str) or not segment_id:
        return [types.TextContent(type='text', text='Error: segment_id is required')]
    segment = store.get(segment_id)
    if segment is None:
        return [types.TextContent(type='text', text=f"Error: No segment found with ID '{segment_id}'")]
    from .write_policy import agent_target_error
    authority_error = agent_target_error(segment)
    if authority_error:
        return [types.TextContent(type='text', text=f'BLOCKED: {authority_error}')]
    if store.is_recycled(segment_id):
        return [types.TextContent(type='text', text='Memory is already in the recoverable recycle bin.')]
    if store.recycle(segment_id, by='agent', reason='Recycled via MCP request') is None:
        return [types.TextContent(type='text', text=f"Error: Failed to recycle segment '{segment_id}'")]
    return [types.TextContent(
        type='text',
        text=f'Recycled memory segment\nID: {segment_id}\nIt is in the recoverable recycle bin; restore it from the dashboard.',
    )]


def get_store():
    store = bound_store()
    if store is None:
        raise RuntimeError('MCP storage must be bound by the explicit SQLite service.')
    return store






def cli():
    """Compatibility alias for the explicitly configured SQLite command."""
    from .sqlite_service import cli as sqlite_cli
    return sqlite_cli()


if __name__ == "__main__":
    cli()
