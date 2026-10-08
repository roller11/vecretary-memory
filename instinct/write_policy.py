# SPDX-License-Identifier: MPL-2.0
"""Owner-only mandatory-memory boundary for agent-facing writes."""

from .models import MANDATORY_TYPES, SegmentType


MAX_AGENT_SCORE = 92


def agent_target_error(segment):
    """Owner records stay protected when nudge temporarily lowers their score."""
    if segment is None:
        return None
    score = getattr(segment, 'importance_score', None)
    initial = getattr(segment, 'initial_score', None)
    segment_type = getattr(getattr(segment, 'segment', None), 'segment_type', None)
    if type(score) is not int or type(initial) is not int or not isinstance(segment_type, SegmentType):
        return 'Cannot confirm target write authority.'
    if score > MAX_AGENT_SCORE or initial > MAX_AGENT_SCORE or segment_type in MANDATORY_TYPES:
        return 'Owner-only records cannot be modified by ordinary agent tools.'
    return None


def agent_write_error(score, segment_type='context'):
    """Reject direct scores and mandatory types that exceed agent authority."""
    if type(score) is not int or not 0 <= score <= 100:
        return 'Importance must be an integer from 0 to 100.'
    if score > MAX_AGENT_SCORE:
        return f'Agent writes are capped at score {MAX_AGENT_SCORE}; higher scores are owner-only.'
    try:
        selected_type = SegmentType(segment_type)
    except (TypeError, ValueError):
        return 'Unknown segment type.'
    if selected_type in MANDATORY_TYPES:
        return 'Mandatory segment types are owner-only; use the reviewed starter command or dashboard.'
    return None
