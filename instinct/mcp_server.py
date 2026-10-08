# SPDX-License-Identifier: MPL-2.0
"""Vecretary Memory stdio MCP server with existing memory write guards."""


import os
os.environ["TQDM_DISABLE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import asyncio
import logging
import sys
import json
import math
import threading
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path
from .privacy_logging import MCPPrivacyFormatter
from .service_binding import bound_store, run_store_call


_stderr_handler = logging.StreamHandler(sys.stderr)
_stderr_handler.setFormatter(MCPPrivacyFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[_stderr_handler])
logger = logging.getLogger("instinct-mcp")


try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp import types
except ImportError as e:
    logger.error("MCP SDK not installed")
    logger.error("Install with: pip install mcp")
    sys.exit(1)













def _probe_embedding_readiness() -> dict:
    """Check the configured embedding provider with a small real embedding."""
    try:
        from .embedding_provider import get_embedding_provider
        provider = get_embedding_provider()
        if provider is None:
            return {"ready": False, "error": "No embedding provider selected", "remediation": "Set CE_EMBEDDING_PROVIDERS=onnx or configure Ollama"}
        vectors = provider.embed_query("Instinct embedding readiness")
        from .provider_privacy import valid_query_embedding
        ready = valid_query_embedding(vectors)
        return {"ready": ready, "model_name": getattr(provider, "model_name", provider.name()), "base_url": getattr(provider, "base_url", "local"), "error": "Invalid or empty embedding" if not ready else None}
    except Exception as error:
        return {"ready": False, "error": f"Embedding probe failed ({type(error).__name__}); details withheld", "remediation": "Check embedding provider settings and model availability"}






def _default_persist_dir() -> str:
    """Return the standalone user data directory."""
    from .runtime import default_data_dir
    return default_data_dir()


def get_instinct_store():
    """Get the InstinctStore instance.

    FALLBACK_STORE (removable): re-raise writer-lock failures instead of
    swallowing them. Previously this returned None on ANY init failure,
    which silently bypassed get_store()'s read-only fallback (the except
    block there never saw an exception) and left every tool dead with
    "initialization failed" during a lock outage. Lock failures are
    transient and expected — get_store() handles them by re-initializing
    read-only. All other exceptions still log and return None.
    """
    try:
        from instinct.store import InstinctStore
        persist_dir = _default_persist_dir()
        return InstinctStore(persist_dir=persist_dir)
    except Exception as e:
        if "ChromaWriterLockUnavailable" in type(e).__name__ or "writer lock" in str(e).lower():
            raise
        logger.error("Failed to initialize InstinctStore")
        return None















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






_shared_writer_enabled = False
_backend_lifecycle = ContextVar('instinct_backend_lifecycle', default=None)


@asynccontextmanager
async def _connection_lifespan(connected_server):
    from .mandatory_lifecycle import MandatoryLifecycle
    client = None
    if _shared_writer_enabled:
        from .shared_writer import SharedWriterClient
        client = SharedWriterClient(_default_persist_dir())
    try:
        yield {"mandatory_lifecycle": MandatoryLifecycle(), "shared_writer_client": client}
    finally:
        if client is not None:
            import anyio
            with anyio.CancelScope(shield=True):
                await client.close()


server = Server("instinct", lifespan=_connection_lifespan)


def _connection_lifecycle():
    backend = _backend_lifecycle.get()
    if backend is not None:
        return backend
    try:
        return server.request_context.lifespan_context.get("mandatory_lifecycle")
    except LookupError:
        return None


def _connection_client():
    try:
        return server.request_context.lifespan_context.get('shared_writer_client')
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
            description="Delete a memory segment by ID. Use instinct_search first to find the ID of the segment to delete.",
            inputSchema={
                "type": "object",
                "properties": {
                    "segment_id": {
                        "type": "string",
                        "description": "The ID of the segment to delete"
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

    client = _connection_client()
    if _shared_writer_enabled and client is None:
        return [types.TextContent(type='text', text='Error: Shared-writer connection context is unavailable; no local store was opened.')]
    if client is not None:
        try:
            return await client.call(name, arguments)
        except Exception:
            logger.error('Shared writer request failed: %s', log_name)
            return [types.TextContent(type='text', text='Error: Shared writer unavailable or response unconfirmed. Check private runtime permissions, matching provider settings and writer ownership. Do not retry writes until their persistence is checked; no locks or guards were changed.')]



    try:
        store = bound_store()
        if store is None:
            store = await asyncio.get_event_loop().run_in_executor(None, get_store)
        if not store:
            logger.error("InstinctStore not available - initialization failed")
            return [types.TextContent(
                type="text",
                text="Error: InstinctStore initialization failed. Check server logs for details."
            )]
    except Exception as e:
        logger.error("Store initialization failed on tool call: %s", log_name)
        return [types.TextContent(
            type="text",
            text="Error: Failed to initialize Instinct context store. Check private storage permissions, dependencies and provider settings; exception details withheld."
        )]

    if bound_store() is store:
        if name == 'instinct_delete_memory':
            return [types.TextContent(type='text', text='Error: Hard deletion is disabled in the experimental SQLite service; no queued write was created.')]
        if store._read_only and name in ('instinct_save_memory', 'instinct_update_score'):
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
            if bound_store() is not store or not store._read_only:
                tick_nudge_session(store, idle_minutes=idle_min)
        except Exception:
            logger.error("Session tick failed (non-fatal)")
            if bound_store() is store:
                raise




        try:
            from instinct import tick_turn, apply_nudge_check
            if bound_store() is not store or not store._read_only:
                tick_turn(store)
        except Exception:
            logger.error("Nudge tick failed (non-fatal)")
            if bound_store() is store:
                raise






        if name in ("instinct_get_context", "instinct_get_mandatory") and (bound_store() is not store or not store._read_only):
            try:
                from instinct import apply_nudge_check
                apply_nudge_check(store)
            except Exception:
                logger.error("Nudge check failed (non-fatal)")
                if bound_store() is store:
                    raise









        try:
            import time as _time
            _now = _time.monotonic()
            if bound_store() is None and _now - getattr(call_tool, "_last_fb_sync_check", 0.0) >= 60.0:
                call_tool._last_fb_sync_check = _now





                try:
                    _try_upgrade_to_write(store)
                except Exception:
                    logger.error("Store upgrade check failed (non-fatal)")
                if not getattr(store, "_read_only", True):
                    from instinct.fallback_store import (
                        pending_count, sync_pending,
                    )
                    if pending_count() > 0:
                        _fb_stats = sync_pending(store)
                        if _fb_stats.get("synced") or _fb_stats.get("failed"):
                            logger.info(
                                "Fallback periodic sync completed"
                            )
        except Exception:
            logger.error("Fallback periodic sync failed (non-fatal)")

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
                    text="Error: Mandatory renewal pending: store is read-only. Retry after owner-approved writer handoff; no locks or guards were changed.",
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








    try:
        segment_id = store.add(scored)
    except RuntimeError as ro_err:
        if "read-only" in str(ro_err).lower():
            from instinct.fallback_store import write_pending
            fb_path = write_pending(
                "save_memory",
                persist_dir=getattr(store, "persist_dir", None),
                content=content,
                title=title,
                importance_score=importance,
                segment_type=seg_type_str,
                reasoning="Saved via MCP tool",
            )
            logger.warning(
                "Store read-only; save captured to fallback"
            )
            return [types.TextContent(
                type="text",
                text=(
                    f"⚠ Saved to FALLBACK (store read-only — writer lock held "
                    f"by another process).\n"
                    f"Pending file: {os.path.basename(fb_path)}\n"
                    f"Title: {title}\nImportance: {importance}\n"
                    f"This memory will sync into Instinct automatically when "
                    f"the writer lock returns to MCP."
                )
            )]
        raise



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


async def _handle_delete_memory(store, arguments: dict) -> list[types.TextContent]:
    """Handle instinct_delete_memory tool."""
    segment_id = arguments.get("segment_id", "")

    if not segment_id:
        return [types.TextContent(type="text", text="Error: segment_id is required")]


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
    title = segment.concept_title
    score = segment.importance_score





    try:
        success = store.delete(segment_id)
    except RuntimeError as ro_err:
        if "read-only" in str(ro_err).lower():
            from instinct.fallback_store import write_pending
            fb_path = write_pending(
                "delete_memory",
                persist_dir=getattr(store, "persist_dir", None)
                or _default_persist_dir(),
                delete_id=segment_id,
            )
            logger.warning(
                "Store read-only; delete captured to fallback"
            )
            return [types.TextContent(
                type="text",
                text=(
                    f"⚠ Delete saved to FALLBACK (store read-only).\n"
                    f"Pending file: {os.path.basename(fb_path)}\n"
                    f"ID: {segment_id}\nTitle: {title}\n"
                    f"Will be deleted from Instinct when the writer lock "
                    f"returns to MCP."
                )
            )]
        raise

    if success:
        logger.info("Segment deleted")
        return [types.TextContent(
            type="text",
            text=f"Deleted segment\nID: {segment_id}\nTitle: {title}\nWas score: {score}"
        )]
    else:
        return [types.TextContent(
            type="text",
            text=f"Error: Failed to delete segment '{segment_id}'"
        )]


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




    try:
        success = store.update_score(segment_id, new_score, reason)
    except RuntimeError as ro_err:
        if "read-only" in str(ro_err).lower():
            from instinct.fallback_store import write_pending
            fb_path = write_pending(
                "update_score",
                persist_dir=getattr(store, "persist_dir", None),
                segment_id=segment_id,
                new_score=new_score,
                score_reason=reason,
            )
            logger.warning(
                "Store read-only; update_score captured to fallback"
            )
            return [types.TextContent(
                type="text",
                text=(
                    f"⚠ Score update saved to FALLBACK (store read-only).\n"
                    f"Pending file: {os.path.basename(fb_path)}\n"
                    f"ID: {segment_id}\nTitle: {title}\n"
                    f"Score: {old_score} -> {new_score}\nReason: {reason}\n"
                    f"Will sync when the writer lock returns to MCP."
                )
            )]
        raise

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






_store = None
_store_init_error = None
_store_lock = threading.Lock()




_UPGRADE_RETRY_SECONDS = 60.0


def _try_upgrade_to_write(store) -> bool:
    """Attempt to upgrade a read-only store to write mode.

    FALLBACK_STORE (removable): closes the "stuck read-only" gap. Once
    the store initialized read-only (writer lock held elsewhere), it was
    cached forever — the periodic sync guard skipped it, so pending
    fallback files never replayed until an MCP restart, even after the
    dashboard released the lock via "Back to MCP".

    Strategy: cheap non-blocking probe (is_writer_lock_held). If the
    lock is free, build a fresh WRITE-mode store, swap it in as the
    global, and replay pending fallback files. On any failure the
    original read-only store stays active — this function never raises.

    Returns True if the store was upgraded.
    """
    if not getattr(store, "_read_only", False):
        return False

    persist_dir = getattr(store, "persist_dir", None) or _default_persist_dir()

    try:
        from instinct.chroma_writer_lock import is_writer_lock_held
        if is_writer_lock_held(persist_dir):
            return False
    except Exception:
        return False



    try:
        from instinct.store import InstinctStore
        new_store = InstinctStore(persist_dir=persist_dir, read_only=False)
    except Exception as e:
        logger.warning(
            "Read-only upgrade attempt failed (lock may have been re-taken)"
        )
        return False




    global _store
    _store = new_store
    logger.info("Store upgraded: read-only -> WRITE mode (lock acquired)")


    try:
        from instinct.fallback_store import sync_pending
        sync_stats = sync_pending(new_store, persist_dir=persist_dir)
        if sync_stats.get("synced") or sync_stats.get("failed"):
            logger.info("Fallback sync after upgrade completed")
    except Exception as fb_err:
        logger.warning("Fallback sync after upgrade skipped")

    return True


def get_store():
    """Get the InstinctStore instance (lazy init, cached, thread-safe).

    FAILSAFE: if the writer lock is unavailable (another process holds it),
    falls back to read-only mode instead of returning None. This means the
    MCP server can still serve get_context, get_mandatory, and search calls
    even when the dashboard or API server is the active writer. Write tools
    (save_memory, update_score, delete) will raise a clear error.
    """
    global _store, _store_init_error
    if _store is not None:
        return _store






    if _store_init_error:
        if "writer lock" in _store_init_error.lower() or "ChromaWriterLockUnavailable" in _store_init_error:
            logger.warning(
                "Previous init failure was transient (writer lock). "
                "Retrying store initialization."
            )
            _store_init_error = None
        else:
            logger.error("Store init failed previously")
            return None
    with _store_lock:

        if _store is not None:
            return _store
        if _store_init_error:
            return None
        logger.info("Initializing InstinctStore (first access)...")
        try:
            _store = get_instinct_store()
            logger.info("InstinctStore initialized successfully (read-write)")





            try:
                from instinct.fallback_store import sync_pending
                sync_stats = sync_pending(
                    _store,
                    persist_dir=_default_persist_dir(),
                )
                if sync_stats.get("synced") or sync_stats.get("failed"):
                    logger.info(
                        "Fallback sync on lock-acquire completed"
                    )
            except Exception as fb_err:
                logger.warning("Fallback sync skipped (non-fatal)")
        except Exception as e:


            err_msg = str(e)
            if "writer lock" in err_msg.lower() or "ChromaWriterLockUnavailable" in type(e).__name__:
                logger.warning(
                    "Writer lock unavailable. Falling back to READ-ONLY mode. "
                    "Write tools may queue pending operations; read tools work normally."
                )
                try:
                    from instinct.store import InstinctStore
                    persist_dir = _default_persist_dir()
                    _store = InstinctStore(persist_dir=persist_dir, read_only=True)
                    logger.info("InstinctStore initialized in READ-ONLY mode")
                except Exception as ro_err:
                    _store_init_error = str(ro_err)
                    logger.error("Failed to init read-only store")
                    return None
            else:
                _store_init_error = err_msg
                logger.error("Failed to initialize InstinctStore")
                return None
    return _store






async def main():
    """Run the MCP server with stdio transport."""
    logger.info("Starting Instinct MCP Server")





    try:
        readiness = {"ready": True} if _shared_writer_enabled else await asyncio.get_event_loop().run_in_executor(
            None, _probe_embedding_readiness
        )
        if _shared_writer_enabled:
            logger.info('Shared-writer proxy initialized; provider readiness deferred to backend')
        elif readiness.get("ready"):
            logger.info(
                "Embedding ready"
            )
        else:
            logger.warning(
                "Embedding NOT ready. Check provider settings and model availability. "
                "Save operations will fail until this is resolved."
            )
    except Exception as e:
        logger.warning("Embedding readiness probe failed")

    try:
        async with stdio_server() as (read_stream, write_stream):
            logger.info("MCP stdio transport initialized")
            logger.info("Ready to accept tool calls (store warming up in background)")
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )
    except Exception as e:
        logger.error("MCP server error")
        raise


def cli():
    """Launch the MCP server from the installed console script."""
    import argparse
    parser = argparse.ArgumentParser(description="Vecretary Memory stdio MCP server")
    parser.add_argument("--data-dir", help="Memory data directory (overrides environment)")
    parser.add_argument("--shared-writer", action="store_true", help="Linux/WSL: route chat connections through one private writer backend")
    arguments = parser.parse_args()
    global _shared_writer_enabled
    _shared_writer_enabled = arguments.shared_writer
    if _shared_writer_enabled:
        from .shared_writer import require_linux
        require_linux()
    if arguments.data_dir:
        os.environ["INSTINCT_DATA_DIR"] = arguments.data_dir
    asyncio.run(main())


if __name__ == "__main__":
    cli()
