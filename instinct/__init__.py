# SPDX-License-Identifier: MPL-2.0
"""Vecretary priority memory library, with the compatible instinct package name.

Provides standalone stdio MCP, a loopback dashboard and priority lifecycle rules.
Starter setup requires explicit owner review and a dedicated empty installation.
Legacy Wisdom-named aliases remain for compatibility; premium Wisdom modules
are not included in the core distribution.
"""

from .models import (
    Segment,
    SegmentType,
    ScoredSegment,
    ProjectIdentity,
    RetrievalResult,
    build_memory_identifiers,
    extract_query_identifiers,
    normalize_identifier,
    parse_memory_identifiers,
)
from .scorer import SegmentScorer, score_segment
from .retriever import PriorityRetriever, get_context
from .mirror import ConversationMirror
from .modifiers import (
    ScoreModifier,
    ModifierPipeline,
    QueryContext,
    ScoringResult,
    ExactIdentifierModifier,
    create_default_pipeline,
)
from .decay import DecayModifier, PopModifier, calculate_effective_score, describe_decay_curve
from .dormancy import (
    DormancyConfig,
    BandConfig,
    DecayFunction,
    DecayProfile,
    bump_interval_active_hours_for,
    should_bump_to_surface,
    current_effective_score,
    decay_series,
    bump_series,
    describe_decay as describe_dormancy_curve,
    describe_band,
    get_config as get_dormancy_config,
    update_config as update_dormancy_config,
    update_band as update_dormancy_band,
    tick_active_clock_public,
    now_active,
    configure as configure_dormancy,
    start_ephemeral,
    tick_ephemeral_turn,
)
from .pop import apply_pop, compute_kpis, describe_decay_path, PopResult, SegmentKPIs
from .nudge import (
    tick_turn,
    get_current_turn,
    apply_access_drop,
    apply_nudge_check,
    get_nudge_state,
    nudge_summary,
    get_nudge_interval,
    set_nudge_interval,
    nudge_active,
    set_nudge_active,
    get_last_nudged_turn,
    get_turns_since_access,
    default_interval_for_segment,
    DEFAULT_NUDGE_INTERVAL,
    ELEVATED_SCORE,
)

# Session clock is the canon for all timed effects in the engine. It
# replaces the legacy active_hours_per_real_day conversion. Ticks are
# driven by MCP tool calls (and the dashboard heartbeat / access
# endpoint as secondary triggers); idle wall-clock time contributes
# zero to aging.
from .session_timer import (
    tick_session,
    session_now,
    session_elapsed_hours,
    accumulated_session_hours,
    session_alive,
    state as session_state,
    reset_session,
    SessionState,
    TickResult,
    DEFAULT_SESSION_IDLE_MINUTES,
)

# --- Wisdom aliases ---------------------------------------------------------
# The dormancy + pop + dashboard system is collectively referred to as the
# "Wisdom" module. These aliases let you refer to it under that name without
# breaking any existing `from instinct.dormancy import ...` code paths.
WisdomConfig = DormancyConfig  # alias: same dataclass, two names
WisdomKPIs = SegmentKPIs       # alias: the dashboard / retriever audit shape
WisdomPop = PopResult          # alias: the pop rule result


def _default_persist_dir() -> str:
    """Resolve the default Instinct persist dir without importing a backend."""
    from .runtime import default_data_dir
    return default_data_dir()


# Lazy imports for optional dependencies
def get_mcp_tools():
    """Get MCP tools for VS Code integration."""
    from .mcp_tools import get_instinct_tools, register_instinct_tools
    return get_instinct_tools, register_instinct_tools

def create_api(*args, **kwargs):
    """Create FastAPI application."""
    from .api import create_instinct_api
    return create_instinct_api(*args, **kwargs)

__all__ = [
    # Models
    "Segment",
    "SegmentType",
    "ScoredSegment",
    "ProjectIdentity",
    "RetrievalResult",
    "build_memory_identifiers",
    "extract_query_identifiers",
    "normalize_identifier",
    "parse_memory_identifiers",
    # Scorer
    "SegmentScorer",
    "score_segment",
    # Retriever
    "PriorityRetriever",
    "get_context",
    # Mirror
    "ConversationMirror",
    # Modifier Pipeline
    "ScoreModifier",
    "ModifierPipeline",
    "QueryContext",
    "ScoringResult",
    "ExactIdentifierModifier",
    "create_default_pipeline",
    # Decay & Pop
    "DecayModifier",
    "PopModifier",
    "calculate_effective_score",
    "describe_decay_curve",
    # Dormancy (v1)
    "DormancyConfig",
    "BandConfig",
    "DecayFunction",
    "DecayProfile",
    "bump_interval_active_hours_for",
    "should_bump_to_surface",
    "current_effective_score",
    "decay_series",
    "bump_series",
    "describe_dormancy_curve",
    "describe_band",
    "get_dormancy_config",
    "update_dormancy_config",
    "update_dormancy_band",
    "tick_active_clock_public",
    "now_active",
    "configure_dormancy",
    "start_ephemeral",
    "tick_ephemeral_turn",
    # Pop (v3, 2026-06-15)
    "apply_pop",
    "compute_kpis",
    "describe_decay_path",
    "PopResult",
    "SegmentKPIs",
    # Session clock (canon for all timed effects)
    "tick_session",
    "session_now",
    "session_elapsed_hours",
    "accumulated_session_hours",
    "session_alive",
    "session_state",
    "reset_session",
    "SessionState",
    "TickResult",
    "DEFAULT_SESSION_IDLE_MINUTES",
    # Wisdom aliases (dormancy + pop + dashboard = "Wisdom" module)
    "WisdomConfig",
    "WisdomKPIs",
    "WisdomPop",
    # Factories
    "get_mcp_tools",
    "create_api",
]

__version__ = "2.0.0"
