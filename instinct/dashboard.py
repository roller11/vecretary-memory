# SPDX-License-Identifier: MPL-2.0
"""Local browser dashboard for Vecretary Memory and lifecycle controls."""
from __future__ import annotations

import json
import math
import secrets
import socket
import os
import sys
import html
import base64
import hashlib
import re
import time
import threading
from importlib import resources
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs


from .runtime import default_data_dir
from .service_binding import bound_store


os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from instinct.models import ScoredSegment, Segment, SegmentType, parse_tag_list
from instinct.retriever import PriorityRetriever
import instinct as _instinct_mod

HOST = "127.0.0.1"
MAX_JSON_BYTES = 1024 * 1024
JSON_READ_TIMEOUT_SECONDS = 5.0
MAX_JSON_DEPTH = 64
MAX_JSON_NODES = 10000
MAX_HTTP_WORKERS = 16
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_QUERY_CHARS = 8192
MAX_LIST_ITEMS = 1000


class RequestBoundaryError(ValueError):
    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.status = status


class DashboardHTTPServer(ThreadingHTTPServer):
    request_queue_size = MAX_HTTP_WORKERS

    def __init__(self, server_address, handler_class):
        self.capability_token = secrets.token_urlsafe(32)
        self._workers = threading.BoundedSemaphore(MAX_HTTP_WORKERS)
        super().__init__(server_address, handler_class)

    def process_request(self, request, client_address):
        if not self._workers.acquire(blocking=False):
            try:
                request.settimeout(1)
                request.sendall(b'HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\nRetry-After: 1\r\n\r\n')
                request.shutdown(socket.SHUT_WR)
                request.settimeout(0.05)
                request.recv(65536)
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._workers.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._workers.release()

    def handle_error(self, request, client_address):
        sys.stderr.write('[dashboard] worker failed; details withheld\n')


def _validated_query(query):
    if len(query) > 12288:
        raise RequestBoundaryError('request target exceeds limit', 414)
    try:
        fields = parse_qs(query, keep_blank_values=True, max_num_fields=32)
    except ValueError:
        raise RequestBoundaryError('too many query fields', 400) from None
    allowed = {'id', 'q', 'n', 'limit', 'score', 'profile', 'days', 'scores', 'include_recycled', 'apply_pops', 'track_access'}
    if set(fields) - allowed or any(len(values) != 1 for values in fields.values()):
        raise RequestBoundaryError('unknown or duplicate query field', 400)
    for name, minimum, maximum in [('n', 1, 100), ('limit', 1, MAX_LIST_ITEMS), ('score', 0, 100), ('days', 1, 90)]:
        if name in fields:
            value = fields[name][0]
            if not value.isascii() or not value.isdecimal() or len(value) > 6 or not minimum <= int(value) <= maximum:
                raise RequestBoundaryError(f'{name} outside supported range', 400)
    for name, maximum in [('q', MAX_QUERY_CHARS), ('id', 160), ('scores', 128)]:
        if name in fields and not 1 <= len(fields[name][0]) <= maximum:
            raise RequestBoundaryError(f'{name} outside text limit', 400)
    if 'profile' in fields and fields['profile'][0] not in ('standard', 'short_term_critical', 'ephemeral'):
        raise RequestBoundaryError('unknown decay profile', 400)
    for name in ('include_recycled', 'apply_pops', 'track_access'):
        if name in fields and fields[name][0] not in ('0', '1', 'false', 'true', 'False', 'True'):
            raise RequestBoundaryError('query flag must be boolean', 400)
    return fields


def _validated_post_fields(data):
    for name, maximum in [('title', 4096), ('content', 65536), ('project', 1024), ('q', MAX_QUERY_CHARS),
                          ('id', 160), ('reason', 4096), ('reasoning', 4096), ('by', 128)]:
        if name in data and (not isinstance(data[name], str) or len(data[name]) > maximum):
            raise RequestBoundaryError(f'{name} must be bounded text', 400)
    if 'ids' in data and (not isinstance(data['ids'], list) or not 1 <= len(data['ids']) <= 100 or
                          any(not isinstance(value, str) or not 1 <= len(value) <= 160 for value in data['ids'])):
        raise RequestBoundaryError('ids require 1 to 100 bounded identifiers', 400)
    for name in ('apply_pops', 'track_access', 'dry_run', 'include_recycle', 'active'):
        if name in data and type(data[name]) is not bool:
            raise RequestBoundaryError('operation flags must be booleans', 400)
    if 'matched_keywords' in data and (type(data['matched_keywords']) is not int or not 0 <= data['matched_keywords'] <= 100):
        raise RequestBoundaryError('matched_keywords must be an integer from 0 to 100', 400)
    if 'tags' in data:
        tags = data['tags']
        if isinstance(tags, str):
            if len(tags) > 8192:
                raise RequestBoundaryError('tags exceed text limit', 400)
        elif not isinstance(tags, list) or len(tags) > 100 or any(not isinstance(tag, str) or len(tag) > 256 for tag in tags):
            raise RequestBoundaryError('tags require at most 100 bounded text values', 400)
    for name, allowed in [('q1', ('short', 'long')), ('q2', ('right_now', 'session', 'next_session', 'multiday', 'reference', 'important', 'solid', 'mandatory'))]:
        if name in data and data[name] not in allowed:
            raise RequestBoundaryError('unknown recommendation choice', 400)


PORT = int(os.environ.get("INSTINCT_PORT") or os.environ.get("CE_PORT", "8765"))
PERSIST_DIR = default_data_dir()








































def get_store():
    isolated = bound_store()
    if isolated is None:
        raise RuntimeError('Dashboard storage is available only inside the SQLite service.')
    return isolated


def get_retriever() -> PriorityRetriever:
    return PriorityRetriever(get_store())


def is_read_only() -> bool:
    """Return the role of the explicitly bound SQLite adapter."""
    try:
        return bool(get_store()._read_only)
    except Exception:
        return False













def readiness_check() -> dict:
    """Describe the service-bound SQLite adapter without opening a backend."""
    store = get_store()
    embeddings_bound = store.query_embedder is not None and store.document_embedder is not None
    reasons = [] if embeddings_bound else ['Embedding bindings missing; text operations unavailable']
    return {
        'ok': True, 'ready': True, 'backend': 'sqlite', 'reasons': reasons,
        'checks': {
            'backend': 'sqlite', 'store_count': store.count(),
            'query_bound': store.query_embedder is not None,
            'documents_bound': store.document_embedder is not None,
            'read_only': store._read_only, 'runtime_ready': False,
            'pending_replay_ready': False,
        },
        'mode': 'read_only' if store._read_only else 'write',
        'pid': os.getpid(), 'ts': datetime.now().isoformat() + 'Z',
    }







def admin_reset(include_recycle: bool = False, dry_run: bool = False) -> dict:
    """Refuse legacy reset semantics; SQLite reset requires a reviewed restore workflow."""
    return {
        'ok': False, 'code': 501,
        'error': 'Dashboard reset is not available. Use an offline SQLite backup and initialize a reviewed replacement runtime.',
        'dry_run': dry_run,
        'include_recycle': include_recycle,
    }
































def _seg_to_dict(s: ScoredSegment) -> dict:
    """Serialize a ScoredSegment to a JSON-friendly dict.

    Real ScoredSegment shape (instinct/models.py):
      s.segment.content, s.segment.segment_type, s.segment.timestamp,
      s.segment.metadata
      s.importance_score  (not s.score)
      s.concept_title     (not s.title — title lives here, not on Segment)
      s.memory_identifiers
      s.last_bumped_at, s.bump_count, s.decay_profile
    """
    seg = s.segment
    md = seg.metadata or {}



    try:
        cfg = _instinct_mod.get_dormancy_config()
        from instinct import compute_kpis
        k = compute_kpis(s, cfg)
        kpis_dict = {
            "initial_score": k.initial_score,
            "current_score": k.current_score,
            "overall_score": round(k.overall_score, 2),
            "initial_semantic_score": round(k.initial_semantic_score, 4),
            "active_hours_elapsed": k.active_hours_elapsed,
            "effective_hours_since_pop": k.effective_hours_since_pop,
            "effective_score": k.effective_score,
            "decay_path_description": k.decay_path_description,
            "initial_decay_path": k.initial_decay_path,
            "keyword_hits_total": k.keyword_hits_total,
            "keyword_hits_24h_active": k.keyword_hits_24h_active,
            "accesses_last_24h_active": k.accesses_last_24h_active,
            "access_count_lifetime": k.access_count_lifetime,
            "last_popped_at": k.last_popped_at,
            "pop_anchor_at": k.pop_anchor_at,
            "pop_recent_8h_active": k.pop_recent_8h_active,
            "promotions_applied": k.promotions_applied,


            "pop_count_total": getattr(s, "bump_count", 0),
        }
    except Exception as e:
        kpis_dict = {"error": "kpi calc failed; details withheld"}

    return {
        "id": s.id,
        "title": s.concept_title or "",
        "content": seg.content,
        "score": s.importance_score,
        "segment_type": seg.segment_type.value if hasattr(seg.segment_type, "value") else str(seg.segment_type),
        "tags": parse_tag_list(md.get("tags", "") or ""),
        "memory_identifiers": list(s.memory_identifiers or []),
        "project": md.get("project", ""),
        "created_at": seg.timestamp.isoformat() if getattr(seg, "timestamp", None) else "",
        "last_bumped_at": s.last_bumped_at.isoformat() if s.last_bumped_at else "",
        "bump_count": s.bump_count,
        "decay_profile": s.decay_profile or "standard",
        "char_count": seg.char_count,
        "is_mandatory": s.is_mandatory,

        "recycled": bool(md.get("recycled_at")),
        "recycled_at": md.get("recycled_at", ""),
        "recycled_by": md.get("recycled_by", ""),
        "recycled_reason": md.get("recycled_reason", ""),

        "kpis": kpis_dict,
        "nudge": _instinct_mod.get_nudge_state(s),
        "nudge": _instinct_mod.get_nudge_state(s),
    }


def _err(msg: str, code: int = 400) -> dict:
    return {"error": msg, "code": code}


def _validated_segment_fields(data):
    title = data.get("title")
    content = data.get("content")
    if not isinstance(title, str) or not title.strip() or not isinstance(content, str) or not content.strip():
        raise ValueError("title and content must be nonempty text")
    score = data.get("score", 50)
    if type(score) is not int or not 0 <= score <= 100:
        raise ValueError("score must be an integer from 0 to 100")
    try:
        segment_type = SegmentType(data.get("segment_type", "context"))
    except (TypeError, ValueError):
        raise ValueError("unknown segment type") from None
    profile = data.get("decay_profile", "standard")
    if profile not in ("standard", "short_term_critical", "ephemeral"):
        raise ValueError("unknown decay profile")
    project = data.get("project", "")
    if not isinstance(project, str):
        raise ValueError("project must be text")
    tags = data.get("tags", [])
    if isinstance(tags, str):
        tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
    if not isinstance(tags, list) or len(tags) > 100 or any(not isinstance(tag, str) for tag in tags):
        raise ValueError("tags must be a list of at most 100 text values")
    return title.strip(), content.strip(), score, segment_type, profile, project.strip(), tags


def _validated_config_updates(data, current):
    import math

    def validate(values, defaults):
        if not isinstance(values, dict) or not values or set(values) - set(defaults):
            raise ValueError("unknown or empty configuration fields")
        result = {}
        for key, value in values.items():
            original = defaults[key]
            if isinstance(original, dict):
                result[key] = {**original, **validate(value, original)}
            elif isinstance(original, str):
                if key != "decay_function" or value not in ("linear", "exponential", "half_life"):
                    raise ValueError("unknown decay function")
                result[key] = value
            else:
                if type(value) not in (int, float) or type(value) is float and not math.isfinite(value) or value < 0:
                    raise ValueError("configuration values must be finite nonnegative numbers")
                if value > 1000000:
                    raise ValueError("configuration value exceeds supported resource bound")
                if type(original) is int and type(value) is not int:
                    raise ValueError("integer configuration field requires an integer")
                if (key.endswith("floor") or key.endswith("score") or key.endswith("drop_to") or key == "initial_drop_pts") and value > 100:
                    raise ValueError("score configuration values must be from 0 to 100")
                if value == 0 and (key.endswith("_turns") or key == "session_idle_minutes" or "total_active_hours" in key or key in ("bump_base_active_hours", "bump_decay_per_10") or key.isdecimal()):
                    raise ValueError("duration and timer values must be positive")
                result[key] = value
        return result

    defaults = current.to_dict()
    defaults["stc_hours_by_score"] = {str(key): value for key, value in defaults["stc_hours_by_score"].items()}
    return validate(data, defaults)
























def _build_modifier_window(seg, cfg) -> dict:
    """Return the modifier-window dict for one segment."""
    initial = seg.initial_score
    current = seg.importance_score
    profile = seg.decay_profile or "standard"
    band = (
        "mandatory" if initial >= 99 else
        "plateau" if initial >= 80 else
        "mid" if initial >= 50 else
        "tail"
    )


    if initial == 100:
        drop_to = cfg.mandatory.floor
        drop_hours = cfg.bump_base_active_hours
    elif initial == 95:
        drop_to = cfg.tier_95_drop_to
        drop_hours = cfg.tier_95_drop_active_hours
    elif initial == 94:
        drop_to = cfg.tier_94_drop_to
        drop_hours = cfg.tier_94_drop_active_hours
    elif initial == 90:
        drop_to = cfg.tier_90_drop_to
        drop_hours = cfg.tier_90_drop_active_hours
    elif initial in (91, 92, 93):
        drop_to = 0
        drop_hours = cfg.stc_hours_by_score.get(initial, cfg.stc_default_active_hours)
    else:

        bcfg = getattr(cfg, band)
        drop_to = max(0, initial - bcfg.initial_drop_pts)
        drop_hours = bcfg.initial_drop_active_hours




    quick_drop_gross = round(drop_to - initial)
    after_quick_drop = drop_to




    try:
        from instinct import dormancy as _dorm
        eff = _dorm.current_effective_score(seg, datetime.now())
        eff_int = round(eff)
    except Exception:
        eff_int = current
    decay_gross = round(eff_int - after_quick_drop)
    after_decay = eff_int



    from instinct.pop import _recent_count_in_active_window
    now = datetime.now()
    kw_8h = _recent_count_in_active_window(seg, 8.0, cfg, now, log_key="pop")
    kw_16h = _recent_count_in_active_window(seg, 16.0, cfg, now, log_key="pop")
    kw_24h = _recent_count_in_active_window(seg, 24.0, cfg, now, log_key="pop")
    kw_total = seg.keyword_hits_total or 0




    pop_gross = round(current - after_decay)




    applied_halve = any(
        e.get("kind") == "pop" for e in (seg.keyword_hits_log or [])
    ) and (seg.last_bumped_at and seg.pop_anchor_at)


    frequency_reset_applied = (seg.pop_recent_8h_active or 0) >= 2

    return {
        "initial_score": initial,
        "band": band,
        "decay_profile": profile,
        "sections": [
            {
                "name": "Quick Drop",
                "description": (
                    f"Drop from {initial} to {drop_to} over "
                    f"{drop_hours:g}h active (initial-drop phase)"
                ),
                "gross_effect": quick_drop_gross,
                "gross_effect_abs": abs(quick_drop_gross),
                "resulting": after_quick_drop,
                "active_hours_elapsed": (
                    round(_dorm._active_age(seg, now, cfg), 2)
                ),
                "config": {
                    "drop_to": drop_to,
                    "drop_hours": drop_hours,
                },
            },
            {
                "name": "Decay Function",
                "description": (
                    f"Band '{band}' decay function over elapsed active time"
                ),
                "gross_effect": decay_gross,
                "gross_effect_abs": abs(decay_gross),
                "resulting": after_decay,
                "config": {
                    "band": band,
                    "decay_function": getattr(getattr(cfg, band, None), "decay_function", "?"),
                    "floor": getattr(getattr(cfg, band, None), "floor", "?"),
                },
            },
            {
                "name": "Keyword Pop",
                "description": "Recent keyword matches can pop this segment back up",
                "gross_effect": pop_gross,
                "gross_effect_abs": abs(pop_gross),
                "resulting": current,
                "boxes": {
                    "hits_8h_active": kw_8h,
                    "hits_16h_active": kw_16h,
                    "hits_24h_active": kw_24h,
                    "hits_total": kw_total,
                },
                "flags": {
                    "halve_applied": bool(applied_halve),
                    "reset_applied": bool(frequency_reset_applied),
                },
                "config": {
                    "halve_threshold": 2,
                    "jump_threshold": 3,
                    "frequency_window_h": 8.0,
                },
            },
        ],
        "final_modified_score": current,
    }








def _recommend_score(q1: str, q2: str, cfg) -> dict:
    """q1: 'short' or 'long'. q2: branch-specific option.

    Returns a flat dict with the headline recommendation PLUS the 7
    scorecard boxes (semantic, importance, time_to_quick_drop,
    decay_modifier, kwpop1, kwpop2, kw_mod_initial). The scorecard is
    just a structured projection of the same recommendation — kept on
    the server so the dashboard JS stays dumb.
    """
    stc = cfg.stc_hours_by_score or {91: 3.0, 92: 24.0, 93: 72.0}
    h91 = stc.get(91, 3.0)
    h92 = stc.get(92, 24.0)
    h93 = stc.get(93, 72.0)


    semantic = q1
    importance = 50
    drop_hours = None
    decay_modifier = "standard"
    kwpop1 = 0
    kwpop2 = 0
    kw_mod_initial = 0

    if q1 == "short":

        if q2 == "right_now":
            importance = 80
            decay_modifier = "ephemeral"
            drop_hours = None
            kwpop1 = 0; kwpop2 = 0; kw_mod_initial = 0
            return _wrap_rec(
                semantic=semantic, importance=importance,
                drop_hours=drop_hours, decay_modifier=decay_modifier,
                kwpop1=kwpop1, kwpop2=kwpop2, kw_mod_initial=kw_mod_initial,
                score=80, decay_profile="ephemeral",
                ephemeral=True,
                tier_label="Ephemeral recall",
                reasoning=(
                    f"Surface for {cfg.ephemeral_surface_turns} turns at full visibility, "
                    f"then fade to dormant floor for {cfg.ephemeral_fade_turns} more turns, "
                    f"then expired. Use for: I need this NOW, briefly retrievable, then gone."
                ),
            )
        if q2 == "session":
            importance = 91
            decay_modifier = "short_term_critical"
            drop_hours = h91
            kwpop1 = 1; kwpop2 = 0; kw_mod_initial = 0
            return _wrap_rec(
                semantic=semantic, importance=importance,
                drop_hours=drop_hours, decay_modifier=decay_modifier,
                kwpop1=kwpop1, kwpop2=kwpop2, kw_mod_initial=kw_mod_initial,
                score=91, decay_profile="short_term_critical",
                tier_label=f"STC session (fades in {h91:g}h active)",
                reasoning=(
                    f"Relevant for this working session. Fades to 0 (retriever hides) "
                    f"after {h91:g} active hours of no access. Unflag to drop immediately."
                ),
            )
        if q2 == "next_session":
            importance = 92
            decay_modifier = "short_term_critical"
            drop_hours = h92
            kwpop1 = 1; kwpop2 = 0; kw_mod_initial = 5
            return _wrap_rec(
                semantic=semantic, importance=importance,
                drop_hours=drop_hours, decay_modifier=decay_modifier,
                kwpop1=kwpop1, kwpop2=kwpop2, kw_mod_initial=kw_mod_initial,
                score=92, decay_profile="short_term_critical",
                tier_label=f"STC next-session (fades in {h92:g}h active)",
                reasoning=(
                    f"Relevant for the next session too. Fades to 0 after "
                    f"{h92:g} active hours of no access. Unflag to drop immediately."
                ),
            )
        if q2 == "multiday":
            importance = 93
            decay_modifier = "short_term_critical"
            drop_hours = h93
            kwpop1 = 1; kwpop2 = 0; kw_mod_initial = 5
            return _wrap_rec(
                semantic=semantic, importance=importance,
                drop_hours=drop_hours, decay_modifier=decay_modifier,
                kwpop1=kwpop1, kwpop2=kwpop2, kw_mod_initial=kw_mod_initial,
                score=93, decay_profile="short_term_critical",
                tier_label=f"STC multi-day (fades in {h93:g}h active)",
                reasoning=(
                    f"Relevant for a few days. Fades to 0 after {h93:g} active hours "
                    f"of no access. Unflag to drop immediately."
                ),
            )

        return _recommend_score("short", "session", cfg)


    if q2 == "reference":
        importance = 95
        decay_modifier = "standard"
        drop_hours = cfg.tier_95_drop_active_hours
        kwpop1 = 1; kwpop2 = 0; kw_mod_initial = 5
        return _wrap_rec(
            semantic=semantic, importance=importance,
            drop_hours=drop_hours, decay_modifier=decay_modifier,
            kwpop1=kwpop1, kwpop2=kwpop2, kw_mod_initial=kw_mod_initial,
            score=95, decay_profile="standard",
            tier_label="Long-term reference (95)",
            reasoning=(
                f"Quick drop to {cfg.tier_95_drop_to} over {cfg.tier_95_drop_active_hours:g}h active, "
                f"then slow linear decay to floor {cfg.tier_95_floor} over "
                f"{cfg.tier_95_total_active_hours:g}h active total. "
                f"Use for: I want this around for a long time, slowly fading."
            ),
        )
    if q2 == "important":
        importance = 94
        decay_modifier = "standard"
        drop_hours = cfg.tier_94_drop_active_hours
        kwpop1 = 1; kwpop2 = 0; kw_mod_initial = 5
        return _wrap_rec(
            semantic=semantic, importance=importance,
            drop_hours=drop_hours, decay_modifier=decay_modifier,
            kwpop1=kwpop1, kwpop2=kwpop2, kw_mod_initial=kw_mod_initial,
            score=94, decay_profile="standard",
            tier_label="Important context (94)",
            reasoning=(
                f"Quick drop to {cfg.tier_94_drop_to} over {cfg.tier_94_drop_active_hours:g}h active, "
                f"then slow linear decay to floor {cfg.tier_94_floor} over "
                f"{cfg.tier_94_total_active_hours:g}h active total. "
                f"Use for: important context I want to recall for weeks/months."
            ),
        )
    if q2 == "solid":
        importance = 90
        decay_modifier = "standard"
        drop_hours = cfg.tier_90_drop_active_hours
        kwpop1 = 1; kwpop2 = 0; kw_mod_initial = 0
        return _wrap_rec(
            semantic=semantic, importance=importance,
            drop_hours=drop_hours, decay_modifier=decay_modifier,
            kwpop1=kwpop1, kwpop2=kwpop2, kw_mod_initial=kw_mod_initial,
            score=90, decay_profile="standard",
            tier_label="Solid context (90)",
            reasoning=(
                f"Quick drop to {cfg.tier_90_drop_to} over {cfg.tier_90_drop_active_hours:g}h active, "
                f"then slow linear decay to floor {cfg.tier_90_floor} over "
                f"{cfg.tier_90_total_active_hours:g}h active total. "
                f"Use for: standard long-term context, comes and goes with relevance."
            ),
        )
    if q2 == "mandatory":
        importance = 100
        decay_modifier = "standard"
        drop_hours = None
        kwpop1 = 1; kwpop2 = 1; kw_mod_initial = 10
        return _wrap_rec(
            semantic=semantic, importance=importance,
            drop_hours=drop_hours, decay_modifier=decay_modifier,
            kwpop1=kwpop1, kwpop2=kwpop2, kw_mod_initial=kw_mod_initial,
            score=100, decay_profile="standard",
            tier_label="Mandatory standing instruction (100)",
            reasoning=(
                f"Always-in-context mandatory. Surfaces every turn at full text, "
                f"with a timer bump every {cfg.bump_base_active_hours:g}h active. "
                f"Use sparingly for non-negotiable rules and standing instructions."
            ),
        )

    return _recommend_score("long", "solid", cfg)


def _wrap_rec(semantic, importance, drop_hours, decay_modifier,
              kwpop1, kwpop2, kw_mod_initial, **core) -> dict:
    """Combine the 7 scorecard boxes with the existing recommendation payload.

    Keeps the original keys (score, decay_profile, tier_label, reasoning,
    ephemeral) intact so the existing /score helper page still works.
    """
    scorecard = {
        "semantic": semantic,
        "importance": importance,
        "time_to_quick_drop_hours": drop_hours,
        "decay_modifier": decay_modifier,
        "kwpop_lvl1": kwpop1,
        "kwpop_lvl2": kwpop2,
        "kw_mod_initial": kw_mod_initial,
    }
    out = dict(core)
    out["scorecard"] = scorecard
    return out




class Handler(BaseHTTPRequestHandler):
    server_version = 'Vecretary'
    sys_version = ''

    def _security_headers(self):
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
        self.send_header('Permissions-Policy', 'camera=(), microphone=(), geolocation=()')

    def log_message(self, fmt, *args):
        sys.stderr.write("[dashboard] request completed\n")

    def setup(self):
        super().setup()
        self.connection.settimeout(JSON_READ_TIMEOUT_SECONDS)

    def _check_boundary(self, *, mutation: bool = False) -> bool:
        if len(self.path) > 12288:
            self._send_json(_err('request target exceeds limit'), 414)
            return False
        request_target = self.requestline.split()[1]
        if not request_target.startswith('/') or request_target.startswith('//') or '\\' in request_target:
            self.close_connection = True
            self._send_json(_err('origin-form request target required'), 400)
            return False
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        host_values = self.headers.get_all("Host", [])
        if len(host_values) != 1 or host_values[0] not in hosts:
            self.close_connection = True
            self._send_json(_err("invalid dashboard Host"), 403)
            return False
        origin_values = self.headers.get_all("Origin", [])
        if origin_values and origin_values != [f"http://{host_values[0]}"]:
            self.close_connection = True
            self._send_json(_err("invalid dashboard Origin"), 403)
            return False
        if self.headers.get("Sec-Fetch-Site", "none") not in ("none", "same-origin"):
            self.close_connection = True
            self._send_json(_err("cross-origin dashboard request denied"), 403)
            return False
        if mutation:
            tokens = self.headers.get_all("X-Instinct-Capability", [])
            if len(tokens) != 1 or not secrets.compare_digest(
                tokens[0].encode("utf-8"), self.server.capability_token.encode("utf-8")
            ):
                self.close_connection = True
                self._send_json(_err("dashboard capability required"), 403)
                return False
        return True

    def _send_json(self, payload, code: int = 200):
        body = bytearray()
        for chunk in json.JSONEncoder(default=str).iterencode(payload):
            encoded = chunk.encode('utf-8')
            if len(body) + len(encoded) > MAX_RESPONSE_BYTES:
                body = b'{"error":"response exceeds limit; use a smaller request"}'
                code = 413
                break
            body.extend(encoded)
        self.send_response(code)
        self._security_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, body: str, code: int = 200):





        if "</header>" in body:
            body = body.replace("</header>", "</header>\n" + MODE_BANNER_HTML, 1)
        elif "<body" in body:

            idx = body.find(">", body.find("<body")) + 1
            body = body[:idx] + "\n" + MODE_BANNER_HTML + body[idx:]
        else:

            body = body + MODE_BANNER_HTML
        bootstrap = """<script>
let instinctReadOnly = true;
let instinctCapability = __INSTINCT_CAPABILITY__;
let instinctCapabilityRefresh = null;
let instinctPreferReadOnly = false;
function dashboardEscapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, character => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[character]));
}
function instinctCanWrite() {
  try { instinctPreferReadOnly = localStorage.getItem('instinct:preferReadOnly') === '1'; } catch {}
  return !instinctReadOnly && !instinctPreferReadOnly;
}
function dashboardApplyWriteState() {
  const blocked = !instinctCanWrite();
  const actions = /^(applyPop|simulateBump|saveAll|runTick|applyRecommendation|ephemeralQuick|onRecycle|onRestore|onPurge|bulkBumpScore|bulkSetScore|bulkRecycle|pullMandatory|saveNudge|runNudge)/;
  document.querySelectorAll('button').forEach(button => {
    if (button.dataset.instinctWrite === '1' || actions.test(button.getAttribute('onclick') || '') || ['save-btn','del-btn','purge-btn','restore-btn','rb-restore-all','rb-empty-bin'].includes(button.id) || button.dataset.action === 'restore' || button.dataset.action === 'purge') {
      button.disabled = blocked;
      button.dataset.instinctWrite = '1';
    }
  });
}
function dashboardRefreshCapability() {
  if (!instinctCapabilityRefresh) {
    instinctCapabilityRefresh = fetch('/api/session', {cache: 'no-store', credentials: 'same-origin', redirect: 'error'}).then(async response => {
      if (!response.ok) throw new Error('Dashboard reconnect failed; reload the page');
      const session = await response.json();
      if (typeof session.capability !== 'string' || !session.capability) throw new Error('Dashboard reconnect failed; reload the page');
      instinctCapability = session.capability;
    }).finally(() => { instinctCapabilityRefresh = null; });
  }
  return instinctCapabilityRefresh;
}
async function dashboardFetch(path, options = {}) {
  const target = new URL(path, location.href);
  if (target.origin !== location.origin) throw new Error('Cross-origin request denied');
  const headers = new Headers(options.headers || {});
  if ((options.method || 'GET').toUpperCase() === 'POST') {
    const readable = ['/api/ready','/api/heartbeat','/api/score/recommend',
      '/api/mode/take_control','/api/mode/return_control'].includes(target.pathname);
    const search = target.pathname === '/api/search_with_pop' && JSON.parse(options.body || '{}').apply_pops === false && JSON.parse(options.body || '{}').track_access === false;
    if (!instinctCanWrite() && !readable && !search) throw new Error('Read-only SQLite observer: restart the SQLite service in owner mode before editing');
    headers.set('X-Instinct-Capability', instinctCapability);
    headers.set('Content-Type', 'application/json');
    options = {...options, body: options.body === undefined ? '{}' : options.body};
  }
  return fetch(path, {...options, headers}).then(async response => {
    if (response.status !== 403 || (options.method || 'GET').toUpperCase() !== 'POST' || target.pathname !== '/api/heartbeat') return response;
    const failure = await response.clone().json().catch(() => ({}));
    if (failure.error !== 'dashboard capability required') return response;
    await dashboardRefreshCapability();
    headers.set('X-Instinct-Capability', instinctCapability);
    return fetch(path, {...options, headers});
  });
}
function dashboardApplySharedTheme() {
  let theme = 'dark';
  try { theme = localStorage.getItem('instinct:theme') === 'light' ? 'light' : 'dark'; } catch {}
  if (theme === 'light') document.body.setAttribute('data-theme', 'light');
  else document.body.removeAttribute('data-theme');
  const button = document.getElementById('theme-toggle');
  if (button) button.textContent = theme === 'light' ? '🌙' : '☀';
}
function dashboardToggleSharedTheme() {
  const current = document.body.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
  const next = current === 'light' ? 'dark' : 'light';
  try { localStorage.setItem('instinct:theme', next); } catch {}
  if (next === 'light') document.body.setAttribute('data-theme', 'light');
  else document.body.removeAttribute('data-theme');
  dashboardApplySharedTheme();
}
function dashboardInstallSharedHeader() {
  const header = document.querySelector('body > header:not(.app-bar)');
  if (!header) return;
  const sourceTitle = header.querySelector('h1');
  const pageName = (sourceTitle?.textContent || document.title).replace(/^Vecretary Memory\\s*[—-]\\s*/, '').trim();
  const retained = [...header.children].filter(child => child !== sourceTitle && child.tagName !== 'A' && !child.matches('div[style*="flex"]'));
  header.classList.add('app-bar', 'secondary-app-bar');
  header.setAttribute('role', 'banner');
  header.replaceChildren();

  const brand = document.createElement('a');
  brand.href = '/';
  brand.className = 'brand';
  brand.title = 'Vecretary Memory Dashboard - home';
  const logo = document.createElement('img');
  logo.className = 'brand-mark';
  logo.src = '/assets/vecretary-logo-256.png';
  logo.alt = 'Vecretary Memory logo';
  brand.appendChild(logo);

  const nav = document.createElement('nav');
  nav.className = 'primary-nav';
  nav.setAttribute('aria-label', 'Primary dashboard');
  const activeHref = location.pathname === '/score' ? '/score' :
    location.pathname === '/decay' ? '/decay' :
    location.pathname === '/modifier' || location.hash === '#segments-section' ? '/#segments-section' : '/';
  for (const [label, href] of [['Overview', '/'], ['Memories', '/#segments-section'], ['Score helper', '/score'], ['Decay', '/decay']]) {
    const link = document.createElement('a');
    link.href = href;
    link.textContent = label;
    if (href === activeHref) link.setAttribute('aria-current', 'page');
    nav.appendChild(link);
  }

  const context = document.createElement('div');
  context.className = 'header-context';
  const title = document.createElement('h1');
  title.className = 'page-title';
  title.textContent = pageName;
  context.appendChild(title);
  const actions = document.createElement('div');
  actions.className = 'header-tools';
  actions.setAttribute('role', 'group');
  actions.setAttribute('aria-label', 'Page actions');
  retained.forEach(element => actions.appendChild(element));
  const themeButton = document.createElement('button');
  themeButton.id = 'theme-toggle';
  themeButton.type = 'button';
  themeButton.setAttribute('aria-label', 'Toggle color theme');
  themeButton.title = 'Toggle light / dark theme';
  themeButton.addEventListener('click', dashboardToggleSharedTheme);
  actions.appendChild(themeButton);
  context.appendChild(actions);
  header.append(brand, nav, context);
}
document.addEventListener('DOMContentLoaded', async () => {
  dashboardInstallSharedHeader();
  dashboardApplySharedTheme();
  const readOnlyPreference = document.getElementById('instinct-prefer-readonly');
  if (readOnlyPreference) {
    try { readOnlyPreference.checked = localStorage.getItem('instinct:preferReadOnly') === '1'; } catch {}
    instinctPreferReadOnly = readOnlyPreference.checked;
    readOnlyPreference.addEventListener('change', () => {
      instinctPreferReadOnly = readOnlyPreference.checked;
      try {
        if (instinctPreferReadOnly) localStorage.setItem('instinct:preferReadOnly', '1');
        else localStorage.removeItem('instinct:preferReadOnly');
      } catch {}
      dashboardApplyWriteState();
    });
    dashboardApplyWriteState();
  }
  async function refreshWriteState() {
    try { const response = await dashboardFetch('/api/mode'); const state = await response.json(); instinctReadOnly = !response.ok || state.read_only; }
    catch { instinctReadOnly = true; }
    dashboardApplyWriteState();
  }
  await refreshWriteState();
  setInterval(refreshWriteState, 4000);
});
</script>""".replace("__INSTINCT_CAPABILITY__", json.dumps(self.server.capability_token))
        responsive = """<style>
:root {
  --bg: #101512;
  --surface: #1a211d;
  --surface-2: #232d26;
  --border: #334239;
  --text: #e5ede7;
  --text-dim: #9aada0;
  --accent: #176331;
  --accent-h: #237943;
  --brand-bright: #5bec7d;
  --brand-deep: #246f3a;
  --success: #1d7136;
  --success-h: #278d46;
  --danger: #a83a2a;
  --danger-h: #c44835;
}
body[data-theme="light"] {
  --bg: #d9dcda;
  --surface: #e6e8e7;
  --surface-2: #d2d7d4;
  --border: #c9d9cd;
  --text: #1a241d;
  --text-dim: #4f6255;
  --accent: #246f3a;
  --accent-h: #287a43;
  --brand-bright: #157a34;
  --brand-deep: #285a38;
  --idle-bg: #e6efe8;
  --idle: #66786a;
  --pill-fg: #405448;
  --success: #1d7136;
  --success-h: #278d46;
  --danger: #a83a2a;
  --danger-h: #c44835;
}
body { color-scheme: dark; background: var(--bg); color: var(--text); }
body[data-theme="light"] { color-scheme: light; }
header { background: var(--surface); border-bottom-color: var(--border); }
header.app-bar { background: color-mix(in srgb, var(--surface) 94%, transparent); }
header a, summary { color: var(--brand-bright); }
main { color: var(--text); }
.panel, .picker, .stage, .section, .status-bar, #seg-list {
  background-color: var(--surface);
  border-color: var(--border);
  color: var(--text);
}
.q, .opt, #scorecard-panel .sc-pill {
  background-color: var(--surface-2);
  border-color: var(--border);
  color: var(--text);
}
.q:hover, .opt:hover, #scorecard-panel .sc-pill:hover { border-color: var(--brand-deep); }
.q.selected, .opt.selected, #scorecard-panel .sc-pill.on {
  background: color-mix(in srgb, var(--accent) 18%, var(--surface));
  border-color: var(--brand-bright);
  color: var(--text);
}
.recommend, #scorecard-panel .sc-box.filled {
  background: color-mix(in srgb, var(--brand-deep) 20%, var(--surface));
  border-color: var(--brand-deep);
}
.recommend h3, .recommend .score, #scorecard-panel .sc-tier,
#scorecard-panel .sc-box.score-stc .v, #scorecard-panel .sc-box.pop-on .v {
  color: var(--brand-bright);
}
.recommend .reasoning, .stage .desc, .stat, .legend { color: var(--text-dim); }
.sc-reasoning, .stage .desc .box, .kb-dialog kbd { background: var(--surface-2); border-color: var(--border); color: var(--text); }
.picker select, .picker input, .stage .gross, .stage .result,
.stage .desc code, .seg-toolbar select, #sort-controls select,
input, textarea, select, fieldset {
  background-color: var(--surface-2);
  border-color: var(--border);
  color: var(--text);
}
button:not(.danger):not(.success) {
  background-color: var(--accent);
  border-color: var(--accent);
  color: #fff;
}
button:not(.danger):not(.success):hover,
.actions button.pop, .actions button.pop:hover {
  background-color: var(--accent-h);
  border-color: var(--accent-h);
  color: #fff;
}
.app-bar nav button:not(.danger):not(.success) {
  background: transparent;
  border-color: transparent;
  color: var(--text-dim);
}
.app-bar nav button:not(.danger):not(.success):hover {
  background: var(--surface-2);
  border-color: var(--border);
  color: var(--text);
}
.actions-menu-panel button:not(.danger):not(.success) {
  background: transparent;
  border-color: transparent;
  color: var(--text);
}
.actions-menu-panel button:not(.danger):not(.success):hover {
  background: var(--surface-2);
  border-color: var(--border);
}
button:focus-visible, a:focus-visible, input:focus-visible,
textarea:focus-visible, select:focus-visible, summary:focus-visible {
  outline: 2px solid var(--brand-bright);
  outline-offset: 2px;
}
a, summary { text-decoration-color: var(--brand-deep); }
.panel h2, .section h2 { color: var(--brand-bright); }
.seg-item.selected { background: color-mix(in srgb, var(--accent) 16%, var(--surface)); border-left-color: var(--brand-bright); }
.seg-item .mod-link, .seg-diff .diff-field { color: var(--brand-bright); }
.seg-item .mod-link { background: color-mix(in srgb, var(--brand-deep) 30%, var(--surface)); border-color: var(--brand-deep); }
.seg-item .mod-link:hover { background: var(--brand-deep); color: #fff; }
.status-bar .path, #sort-controls select { color: var(--brand-bright); }
.status-bar .path { background: var(--surface-2); }
.mode-banner:not(.write):not(.read_only) { background: var(--surface-2); border: 1px solid var(--border); color: var(--text); }
.mode-banner .status, .mode-banner .read-only-preference-note { color: var(--text-dim); }
.toast:not(.error) { background: var(--brand-deep); color: #fff; }
.stage.initial { border-color: var(--brand-deep); }
.stage.final { border-color: var(--brand-bright); }
.stage .gross.pos, .stage .desc .flags .on { color: var(--brand-bright); }
.seg-diff { background: var(--surface-2); border-color: var(--border); color: var(--text); }
.seg-diff .diff-added { background: color-mix(in srgb, var(--brand-deep) 25%, var(--surface)); color: var(--brand-bright); }
.seg-diff .diff-changed { background: color-mix(in srgb, var(--brand-deep) 18%, var(--surface)); color: var(--text); }
#bulk-bar { background: color-mix(in srgb, var(--brand-deep) 24%, var(--surface)); border-color: var(--brand-deep); color: var(--text); }
#bulk-bar b { color: var(--brand-bright); }
#bulk-bar button:not(.danger) { background: var(--brand-deep); border-color: var(--brand-bright); color: #fff; }
.pop-audit { background: color-mix(in srgb, var(--brand-deep) 28%, var(--surface)); border-color: var(--brand-deep); color: var(--brand-bright); }
.sparkline { background: var(--surface-2); border-color: var(--border); }
.score-cta { background: color-mix(in srgb, var(--brand-deep) 28%, var(--surface)); border-color: var(--brand-deep); color: var(--brand-bright); }
.score-cta:hover { background: var(--brand-deep); color: #fff; }
svg { background-color: var(--surface-2); }
@media (max-width: 900px) {
  header {position:static !important; flex-wrap:wrap;}
  main {display:block !important;}
  .picker {flex-wrap:wrap;}
  .picker select, .picker input {min-width:0; max-width:100%; width:100%;}
  .stage {grid-template-columns:1fr 50px 1fr 70px;}
}
:root {
  --control-shadow: 0 2px 4px rgba(0,0,0,.22), inset 0 1px 0 rgba(255,255,255,.14);
  --control-shadow-hover: 0 4px 9px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.20);
  --control-shadow-pressed: inset 0 2px 4px rgba(0,0,0,.24);
}
body[data-theme="light"] {
  --control-shadow: 0 2px 4px rgba(24,50,32,.14), inset 0 1px 0 rgba(255,255,255,.82);
  --control-shadow-hover: 0 4px 9px rgba(24,50,32,.18), inset 0 1px 0 rgba(255,255,255,.94);
  --control-shadow-pressed: inset 0 2px 4px rgba(24,50,32,.16);
}
button,
input[type="button"], input[type="submit"], input[type="reset"],
[role="button"], .btn, .button,
.header-primary, .app-bar nav a, .actions-menu > summary, .score-cta, .mod-link {
  box-sizing: border-box;
  height: max(24px, calc(1lh + 5px));
  min-height: max(24px, calc(1lh + 5px));
  padding-block: 0;
  line-height: inherit;
  display: inline-flex;
  align-items: center;
  background-image: linear-gradient(180deg, rgba(255,255,255,.13), rgba(0,0,0,.07));
  box-shadow: var(--control-shadow);
  transition: background-color .14s ease, border-color .14s ease, box-shadow .14s ease, transform .14s ease, filter .14s ease;
}
button:hover:not(:disabled),
input[type="button"]:hover, input[type="submit"]:hover, input[type="reset"]:hover,
[role="button"]:hover, .btn:hover, .button:hover,
.header-primary:hover, .app-bar nav a:hover, .actions-menu > summary:hover,
.score-cta:hover, .mod-link:hover {
  box-shadow: var(--control-shadow-hover);
  filter: saturate(1.04);
  transform: translateY(-1px);
}
button:active:not(:disabled),
input[type="button"]:active, input[type="submit"]:active, input[type="reset"]:active,
[role="button"]:active, .btn:active, .button:active,
.header-primary:active, .app-bar nav a:active, .actions-menu > summary:active,
.score-cta:active, .mod-link:active {
  box-shadow: var(--control-shadow-pressed);
  filter: none;
  transform: translateY(1px);
}
button:disabled, input[type="button"]:disabled, input[type="submit"]:disabled, input[type="reset"]:disabled {
  background-image: none;
  box-shadow: none;
  filter: none;
  transform: none;
}
@media (prefers-reduced-motion: reduce) {
  button, input[type="button"], input[type="submit"], input[type="reset"],
  [role="button"], .btn, .button, .header-primary, .app-bar nav a,
  .actions-menu > summary, .score-cta, .mod-link {
    transition: none;
    transform: none !important;
  }
}
</style>"""
        surface_layers = """<style>
:root {
  --surface-raised: #202923;
  --surface-inset: #151b17;
  --frame-shadow: 0 3px 12px rgba(0,0,0,.18), inset 0 1px 0 rgba(255,255,255,.035);
  --list-shadow: inset 0 2px 8px rgba(0,0,0,.16), inset 0 1px 0 rgba(255,255,255,.025);
  --decay-score-100: #ffd166;
  --decay-score-90: #ff8f70;
  --decay-score-80: #4cc9f0;
  --decay-score-70: #c8a7ff;
  --decay-score-50: #ff8fab;
  --decay-score-20: #8ce99a;
  --decay-floor-color: #ff6b6b;
  --header-gradient-start: #101a14;
  --header-gradient-mid: #183a25;
  --header-gradient-end: #386543;
}
body[data-theme="light"] {
  --surface-raised: #eceeed;
  --surface-inset: #d2d7d4;
  --frame-shadow: 0 3px 12px rgba(24,50,32,.09), inset 0 1px 0 rgba(255,255,255,.88);
  --list-shadow: inset 0 2px 8px rgba(24,50,32,.07), inset 0 1px 0 rgba(255,255,255,.72);
  --decay-score-100: #9a6700;
  --decay-score-90: #c2412d;
  --decay-score-80: #006d91;
  --decay-score-70: #6741d9;
  --decay-score-50: #b4235a;
  --decay-score-20: #287a3a;
  --decay-floor-color: #b42318;
  --header-gradient-start: #cfd9d2;
  --header-gradient-mid: #dce2de;
  --header-gradient-end: #e6e8e7;
}
body[data-theme="light"] #readonly-banner {
  background: #762727 !important;
  border-bottom-color: #c95a5a !important;
  color: #fff !important;
}
body .panel, body .picker, body .stage, body .section {
  background-image: linear-gradient(145deg, var(--surface-raised), var(--surface) 72%);
  box-shadow: var(--frame-shadow);
}
body .status-bar {
  background-image: linear-gradient(100deg, var(--surface-2), var(--surface-raised));
  box-shadow: inset 0 1px 0 rgba(255,255,255,.035);
}
body #seg-list {
  background-color: var(--surface-inset);
  background-image: linear-gradient(180deg, var(--surface-inset), var(--surface-2));
  box-shadow: var(--list-shadow);
}
body #segments-section #seg-list .seg-item {
  background-color: var(--surface);
  background-image: linear-gradient(105deg, color-mix(in srgb, var(--surface-raised) 70%, transparent), transparent 72%);
  box-shadow: inset 0 1px 0 color-mix(in srgb, var(--text) 4%, transparent);
}
body #segments-section #seg-list .seg-item:nth-child(even) {
  background-color: color-mix(in srgb, var(--surface-2) 24%, var(--surface));
}
body #segments-section #seg-list .seg-item:hover:not(.selected) {
  background-image: linear-gradient(105deg, color-mix(in srgb, var(--accent) 12%, var(--surface-2)), var(--surface) 72%);
}
body #segments-section #seg-list .seg-item.selected {
  background-image: linear-gradient(105deg, color-mix(in srgb, var(--accent) 23%, var(--surface-raised)), color-mix(in srgb, var(--accent) 11%, var(--surface)));
}
body #graph, body #bumpGraph {
  border: 1px solid var(--border);
  box-shadow: inset 0 1px 0 rgba(255,255,255,.05), 0 2px 8px rgba(0,0,0,.12);
}
body #tunables fieldset[data-range] {
  --range-color: var(--decay-score-80);
  box-sizing: border-box;
  min-width: 0;
  margin: 10px 0;
  padding: 10px 10px 6px;
  border: 2px solid var(--range-color);
  border-radius: 8px;
  background: linear-gradient(145deg, color-mix(in srgb, var(--range-color) 7%, var(--surface-2)), var(--surface-2));
}
body #tunables fieldset[data-range="mandatory"] { --range-color: var(--decay-score-100); }
body #tunables fieldset[data-range^="tier-"] { --range-color: var(--decay-score-90); }
body #tunables fieldset[data-range="plateau"] { --range-color: var(--decay-score-80); }
body #tunables fieldset[data-range="mid"] {
  --range-color: var(--decay-score-70);
  box-shadow: inset 0 0 0 1px var(--decay-score-50);
}
body #tunables fieldset[data-range="tail"] { --range-color: var(--decay-score-20); }
body #tunables fieldset[data-range] > legend {
  padding: 0 6px;
  color: var(--range-color);
  font-weight: 700;
}
body #tunables fieldset[data-range] input,
body #tunables fieldset[data-range] select {
  border-color: color-mix(in srgb, var(--range-color) 68%, var(--border));
}
body #tunables fieldset[data-range] input:focus-visible,
body #tunables fieldset[data-range] select:focus-visible { outline-color: var(--range-color); }
body #legend .swatch { box-shadow: 0 0 0 1px var(--border); }
body header.app-bar {
  position: static !important;
  inset-block-start: auto !important;
  z-index: auto;
  display: grid;
  grid-template-columns: minmax(340px, 1fr) minmax(0, 2fr);
  grid-template-rows: minmax(0, 1fr) minmax(0, 1fr);
  align-items: center;
  gap: 12px 24px;
  min-height: 280px;
  height: 280px;
  max-height: 280px;
  padding: 10px 16px;
  box-sizing: border-box;
  background-color: var(--header-gradient-start);
  background-image: linear-gradient(135deg, var(--header-gradient-start) 0%, var(--header-gradient-mid) 58%, var(--header-gradient-end) 100%);
  border-bottom: 1px solid color-mix(in srgb, var(--header-gradient-end) 42%, var(--border));
  box-shadow: 0 7px 22px rgba(0,0,0,.22), inset 0 1px 0 rgba(255,255,255,.07);
  backdrop-filter: none;
}
body header.app-bar > .brand {
  grid-column: 1;
  grid-row: 1 / span 2;
  display: flex;
  align-items: center;
  gap: 14px;
  min-height: 256px;
  white-space: nowrap;
}
body header.app-bar .brand-mark {
  display: block;
  width: 256px;
  height: 256px;
  flex: 0 0 256px;
  object-fit: contain;
  padding: 0;
  background: transparent;
  border: 0;
  border-radius: 12px;
  filter: drop-shadow(0 4px 12px rgba(0,0,0,.28));
}
body header.app-bar .brand-wordmark { font-size: 20px; font-weight: 700; }
body header.app-bar .primary-nav {
  grid-column: 2;
  grid-row: 1;
  display: flex;
  flex-basis: auto;
  flex-wrap: wrap;
  justify-content: flex-end;
  align-content: end;
  min-width: 0;
  overflow: visible;
  padding-bottom: 0;
  order: 0;
}
body header.secondary-app-bar .primary-nav { gap: 5px; }
body header.secondary-app-bar .primary-nav a {
  color: var(--text-dim);
  text-decoration: none;
  display: inline-flex;
  align-items: center;
  padding-inline: 13px;
  border: 1px solid var(--border);
  border-radius: 10px;
  background-color: var(--surface-2);
  font-size: 14px;
  font-weight: 550;
  font-family: inherit;
  white-space: nowrap;
  cursor: pointer;
}
body header.secondary-app-bar .primary-nav a:hover {
  background-color: var(--surface-raised);
  color: var(--text);
}
body header.app-bar .primary-nav a[aria-current="page"] {
  color: var(--text);
  background: color-mix(in srgb, var(--header-gradient-end) 28%, transparent);
  border-color: color-mix(in srgb, var(--header-gradient-end) 65%, var(--border));
}
body header.app-bar > .header-tools {
  grid-column: 2;
  grid-row: 2;
  justify-self: end;
  align-self: start;
  width: auto;
  margin: 0;
  order: 0;
}
body header.app-bar #theme-toggle {
  width: max(24px, calc(1lh + 5px));
  min-width: max(24px, calc(1lh + 5px));
  height: max(24px, calc(1lh + 5px));
  min-height: max(24px, calc(1lh + 5px));
  padding: 0;
  border-radius: 10px;
}
body header.app-bar #recycle-link {
  box-sizing: border-box;
  width: max(24px, calc(1lh + 5px));
  min-width: max(24px, calc(1lh + 5px));
  height: max(24px, calc(1lh + 5px));
  min-height: max(24px, calc(1lh + 5px));
  padding: 0;
  justify-content: center;
  border-radius: 10px;
  font-size: 14px;
}
body header.secondary-app-bar .header-context {
  grid-column: 2;
  grid-row: 2;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px 18px;
  min-width: 0;
  align-self: start;
}
body header.secondary-app-bar .page-title {
  margin: 0;
  color: var(--text);
  font-size: 18px;
  font-weight: 650;
}
body header.secondary-app-bar .header-tools {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: flex-end;
  margin: 0;
}
@media (max-width: 900px) {
  body header.app-bar {
    grid-template-columns: minmax(0, 256px) minmax(0, 1fr);
    grid-template-rows: minmax(0, 1fr) minmax(0, 1fr);
    padding: 10px 12px;
    gap: 8px 14px;
  }
  body header.app-bar > .brand {
    grid-column: 1;
    grid-row: 1 / span 2;
    flex-direction: row;
    justify-content: center;
    gap: 10px;
    min-height: 256px;
    white-space: nowrap;
  }
  body header.app-bar .brand-mark {
    width: 256px;
    height: 256px;
    flex-basis: 256px;
  }
  body header.app-bar .primary-nav {
    grid-column: 2;
    grid-row: 1;
    justify-content: flex-end;
    align-content: center;
    flex-wrap: wrap;
    overflow: visible;
    max-width: 100%;
  }
  body header.app-bar > .header-tools {
    grid-column: 2;
    grid-row: 2;
    justify-self: stretch;
    align-self: center;
    width: 100%;
    min-width: 0;
    flex-wrap: wrap;
    justify-content: flex-end;
  }
  body header.secondary-app-bar .header-context {
    grid-column: 2;
    grid-row: 2;
    flex-direction: row;
    justify-content: space-between;
    text-align: left;
    flex-wrap: wrap;
    min-width: 0;
  }
  body header.secondary-app-bar .header-tools { justify-content: flex-end; }
}
@media (max-width: 520px) {
  body header.app-bar {
    grid-template-columns: minmax(0, 1fr);
    grid-template-rows: 112px 42px minmax(0, 1fr);
    padding: 8px 10px;
    gap: 4px;
  }
  body header.app-bar > .brand {
    grid-column: 1;
    grid-row: 1;
    min-height: 0;
  }
  body header.app-bar .brand-mark {
    width: 112px;
    height: 112px;
    flex-basis: 112px;
  }
  body header.app-bar .primary-nav {
    grid-column: 1;
    grid-row: 2;
    justify-content: flex-start;
    flex-wrap: nowrap;
    overflow-x: auto;
    overflow-y: hidden;
    padding-bottom: 0;
  }
  body header.app-bar .primary-nav a,
  body header.app-bar .primary-nav button {
    height: 34px;
    min-height: 34px;
    padding-inline: 10px;
  }
  body header.app-bar > .header-tools {
    grid-column: 1;
    grid-row: 3;
    justify-self: stretch;
    width: 100%;
    justify-content: flex-start;
  }
  body header.app-bar .header-primary,
  body header.app-bar #theme-toggle,
  body header.app-bar .actions-menu > summary {
    height: 34px;
    min-height: 34px;
    padding-inline: 10px;
  }
  body header.app-bar #recycle-link {
    width: 34px;
    min-width: 34px;
    height: 34px;
    min-height: 34px;
    padding: 0;
  }
  body header.app-bar #theme-toggle {
    width: 34px;
    min-width: 34px;
    padding: 0;
  }
  body header.secondary-app-bar .header-context {
    grid-column: 1;
    grid-row: 3;
    flex-direction: column;
    align-items: flex-start;
    justify-content: flex-start;
    gap: 4px;
  }
  body header.secondary-app-bar .header-tools {
    width: 100%;
    justify-content: flex-start;
  }
}
</style>"""
        body = body.replace("<head>", "<head>" + bootstrap, 1)
        body = body.replace("</head>", responsive + "</head>", 1)
        body = body.replace("</body>", surface_layers + "</body>", 1)
        nonce = secrets.token_urlsafe(24)
        handlers = set(re.findall(r'on(?:click|submit)="([^"]+)"', body))
        hashes = ' '.join("'sha256-" + base64.b64encode(hashlib.sha256(html.unescape(handler).encode()).digest()).decode() + "'" for handler in sorted(handlers))
        body = body.replace('<script>', f'<script nonce="{nonce}">')
        policy = f"default-src 'none'; script-src 'nonce-{nonce}'; script-src-attr 'unsafe-hashes' {hashes}; style-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'self'; frame-ancestors 'none'; object-src 'none'"
        b = body.encode("utf-8")
        self.send_response(code)
        self._security_headers()
        self.send_header('Content-Security-Policy', policy)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def _read_json(self) -> dict:
        if self.headers.get_all("Transfer-Encoding", []):
            raise RequestBoundaryError("transfer encoding is not supported", 400)
        content_types = self.headers.get_all("Content-Type", [])
        if len(content_types) != 1 or content_types[0].split(";", 1)[0].strip().lower() != "application/json":
            raise RequestBoundaryError("application/json required", 415)
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit():
            raise RequestBoundaryError("one valid Content-Length required", 400)
        if len(lengths[0]) > 10 or int(lengths[0]) > MAX_JSON_BYTES:
            raise RequestBoundaryError("JSON body exceeds size limit", 413)
        length = int(lengths[0])
        if length <= 0:
            raise RequestBoundaryError("JSON object required", 400)
        chunks = []
        remaining = length
        deadline = time.monotonic() + JSON_READ_TIMEOUT_SECONDS
        try:
            while remaining:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    raise RequestBoundaryError("JSON body read timed out", 408)
                self.connection.settimeout(timeout)
                chunk = self.rfile.read1(min(remaining, 65536))
                if not chunk:
                    raise RequestBoundaryError("incomplete JSON body", 400)
                chunks.append(chunk)
                remaining -= len(chunk)
        except (TimeoutError, socket.timeout):
            raise RequestBoundaryError("JSON body read timed out", 408) from None
        finally:
            self.connection.settimeout(JSON_READ_TIMEOUT_SECONDS)
        try:
            data = json.loads(b"".join(chunks).decode("utf-8"), parse_constant=self._reject_json_constant)
        except (ValueError, UnicodeError, RecursionError):
            raise RequestBoundaryError("invalid JSON object", 400) from None
        if not isinstance(data, dict):
            raise RequestBoundaryError("JSON object required", 400)
        pending = [(data, 0)]
        visited = 0
        while pending:
            value, depth = pending.pop()
            visited += 1
            if depth > MAX_JSON_DEPTH or visited > MAX_JSON_NODES:
                raise RequestBoundaryError("JSON structure exceeds limit", 400)
            if isinstance(value, float) and not math.isfinite(value):
                raise RequestBoundaryError("non-finite JSON number", 400)
            if isinstance(value, dict):
                pending.extend((child, depth + 1) for child in value.values())
            elif isinstance(value, list):
                pending.extend((child, depth + 1) for child in value)
        return data

    @staticmethod
    def _reject_json_constant(value):
        raise ValueError("non-finite JSON number")

    def do_GET(self):
        if not self._check_boundary():
            return
        url = urlparse(self.path)
        try:
            _validated_query(url.query)
            if url.path == "/api/session":
                return self._send_json({"capability": self.server.capability_token})
            if url.path == "/" or url.path == "/index.html":
                return self._send_html(INDEX_HTML)
            if url.path in ("/assets/vecretary-logo.png", "/assets/vecretary-logo-256.png"):
                logo_name = "vecretary-logo-256.png" if url.path == "/assets/vecretary-logo-256.png" else "vecretary-logo-512.png"
                image = resources.files("instinct").joinpath(
                    "assets", logo_name).read_bytes()
                self.send_response(200)
                self._security_headers()
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(image)))
                self.send_header("Cache-Control", "public, max-age=86400")
                self.end_headers()
                self.wfile.write(image)
                return
            if url.path == "/favicon.ico":

                gif = b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\x00\x00\x00!\xf9\x04\x00\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"
                self.send_response(200)
                self.send_header("Content-Type", "image/gif")
                self.send_header("Content-Length", str(len(gif)))
                self.send_header("Cache-Control", "public, max-age=86400")
                self.end_headers()
                self.wfile.write(gif)
                return
            if url.path == "/api/health":
                store = get_store()
                ready = readiness_check()
                return self._send_json({
                    "ok": True,
                    "ready": ready["ok"],
                    "degraded_reasons": ready["reasons"],
                    "backend": "sqlite",
                    "store_dir": str(store.persist_dir),
                    "writer_lock": {"held_by_us": not store._read_only},
                    "read_only": is_read_only(),
                    "pid": os.getpid(),
                    "embedding_model": store.repository.identity.model,
                    "runtime_ready": False,
                    "ts": datetime.now().isoformat() + "Z",
                })
            if url.path == "/api/ready":



                report = readiness_check()
                code = 200 if report["ok"] else 503
                return self._send_json(report, code)
            if url.path == "/api/mode":
                store = get_store()
                return self._send_json({
                    "backend": "sqlite",
                    "read_only": store._read_only,
                    "mode": "read_only" if store._read_only else "write",
                    "handoff_supported": False,
                    "runtime_ready": False,
                    "pid": os.getpid(),
                })
            if url.path == "/api/stats":
                return self._send_json(get_store().stats())
            if url.path == "/api/list":
                qs = parse_qs(url.query)
                limit = int(qs.get("limit", ["200"])[0])
                include_recycled = qs.get("include_recycled", ["0"])[0] not in ("0", "false", "False")
                store = get_store()
                items = [_seg_to_dict(s) for s in store.get_by_score(
                    min_score=0, limit=limit, include_recycled=include_recycled,
                )]
                return self._send_json({"count": store.count(), "items": items})
            if url.path == "/api/recycled":


                items = [_seg_to_dict(s) for s in get_store().list_recycled(limit=500)]
                return self._send_json({"count": len(items), "items": items})
            if url.path == "/api/mandatory":
                items = [_seg_to_dict(s) for s in get_store().get_mandatory()]
                return self._send_json({"count": len(items), "items": items})
            if url.path == "/api/nudge/summary":

                from instinct import nudge_summary, get_current_turn
                return self._send_json(nudge_summary(get_store()))
            if url.path == "/api/nudge/list":

                from instinct import get_nudge_state
                items = [get_nudge_state(s) for s in get_store().get_mandatory()]
                return self._send_json({"count": len(items), "items": items})
            if url.path == "/api/turn":

                from instinct import get_current_turn
                return self._send_json({"current_turn": get_current_turn(get_store())})
            if url.path == "/api/get":
                qs = parse_qs(url.query)
                seg_id = qs.get("id", [""])[0]
                if not seg_id:
                    return self._send_json(_err("missing id"), 400)
                seg = get_store().get(seg_id)
                if not seg:
                    return self._send_json(_err("not found", 404), 404)
                return self._send_json(_seg_to_dict(seg))
            if url.path == "/api/segment/kpis":

                qs = parse_qs(url.query)
                seg_id = qs.get("id", [""])[0]
                if not seg_id:
                    return self._send_json(_err("missing id"), 400)
                seg = get_store().get(seg_id)
                if not seg:
                    return self._send_json(_err("not found", 404), 404)
                from instinct import compute_kpis
                cfg = _instinct_mod.get_dormancy_config()
                k = compute_kpis(seg, cfg)
                return self._send_json({
                    "id": seg_id,
                    "kpis": {
                        "initial_score": k.initial_score,
                        "current_score": k.current_score,
                        "overall_score": round(k.overall_score, 2),
                        "initial_semantic_score": round(k.initial_semantic_score, 4),
                        "initial_decay_path": k.initial_decay_path,
                        "active_hours_elapsed": k.active_hours_elapsed,
                        "effective_hours_since_pop": k.effective_hours_since_pop,
                        "effective_score": k.effective_score,
                        "decay_path_description": k.decay_path_description,
                        "keyword_hits_total": k.keyword_hits_total,
                        "keyword_hits_24h_active": k.keyword_hits_24h_active,
                        "accesses_last_24h_active": k.accesses_last_24h_active,
                        "access_count_lifetime": k.access_count_lifetime,
                        "last_bumped_at": k.last_bumped_at,
                        "last_popped_at": k.last_popped_at,
                        "pop_anchor_at": k.pop_anchor_at,
                        "pop_recent_8h_active": k.pop_recent_8h_active,
                        "promotions_applied": k.promotions_applied,
                        "is_mandatory": k.is_mandatory,
                        "decay_profile": k.decay_profile,
                    },
                })
            if url.path == "/api/segment/pop":


                pass
            if url.path == "/api/segment/modifier_window":

                qs = parse_qs(url.query)
                seg_id = qs.get("id", [""])[0]
                if not seg_id:
                    return self._send_json(_err("missing id"), 400)
                seg = get_store().get(seg_id)
                if not seg:
                    return self._send_json(_err("not found", 404), 404)
                cfg = _instinct_mod.get_dormancy_config()
                try:
                    window = _build_modifier_window(seg, cfg)
                    return self._send_json({
                        "id": seg_id,
                        "title": seg.concept_title or "",
                        "modifier_window": window,
                    })
                except Exception as e:
                    sys.stderr.write('[dashboard] optional operation failed; details withheld\n')
                    return self._send_json(_err('modifier build failed; details withheld'), 500)
            if url.path == "/modifier":

                return self._send_html(MODIFIER_HTML)
            if url.path == "/api/search":
                qs = parse_qs(url.query)
                q = qs.get("q", [""])[0]
                n = int(qs.get("n", ["10"])[0])
                include_recycled = qs.get("include_recycled", ["0"])[0] not in ("0", "false", "False")
                if not q:
                    return self._send_json(_err("missing q"), 400)

                results = get_store().search(query=q, limit=n, include_recycled=include_recycled)
                items = [_seg_to_dict(s) for s in results]
                return self._send_json({"count": len(items), "items": items})
            if url.path == "/api/search_with_pop":


                qs = parse_qs(url.query)
                q = qs.get("q", [""])[0]
                n = int(qs.get("n", ["10"])[0])
                apply_pops = qs.get("apply_pops", ["0"])[0] not in ("0", "false", "False")
                track_access = qs.get("track_access", ["0"])[0] not in ("0", "false", "False")
                if apply_pops or track_access:
                    return self._send_json(_err("mutating search requires POST"), 405)
                if not q:
                    return self._send_json(_err("missing q"), 400)
                retriever = get_retriever()
                result = retriever.retrieve(
                    q, max_items=n, apply_pops=apply_pops, track_access=track_access,
                )
                return self._send_json({
                    "query": q,
                    "applied_pops": apply_pops,
                    "tracked_access": track_access,
                    "mandatory_count": len(result.mandatory),
                    "relevant_count": len(result.relevant),
                    "pop_count": len(result.pop_results),
                    "mandatory": [_seg_to_dict(s) for s in result.mandatory],
                    "relevant": [_seg_to_dict(s) for s in result.relevant],
                    "pop_results": [
                        {
                            "segment_id": pr.segment_id,
                            "segment_title": pr.segment_title,
                            "matched_keywords": pr.matched_keywords,
                            "rule_applied": pr.rule_applied,
                            "initial_score_before": pr.initial_score_before,
                            "initial_score_after": pr.initial_score_after,
                            "effective_score_before": pr.effective_score_before,
                            "effective_score_after": pr.effective_score_after,
                            "anchor_reset": pr.anchor_reset,
                            "was_mandatory": pr.was_mandatory,
                            "message": pr.message,
                        }
                        for pr in result.pop_results
                    ],
                })



            if url.path == "/decay" or url.path == "/decay.html":
                return self._send_html(DECAY_HTML)
            if url.path == "/score" or url.path == "/score.html":
                return self._send_html(SCORE_HTML)
            if url.path == "/api/decay/config":
                from instinct import get_dormancy_config
                from instinct import session_timer as _st
                cfg = get_dormancy_config().to_dict()



                sstate = _st.state()
                cfg["session_state"] = {
                    "session_started_at": sstate.session_started_at,
                    "session_last_tick_at": sstate.session_last_tick_at,
                    "accumulated_session_hours": round(
                        sstate.accumulated_session_seconds / 3600.0, 4
                    ),
                    "session_alive": _st.session_alive(
                        idle_minutes=float(cfg.get("session_idle_minutes", 10.0))
                    ),
                }
                return self._send_json(cfg)
            if url.path == "/api/decay/series":
                qs = parse_qs(url.query)
                score = int(qs.get("score", ["100"])[0])
                profile = qs.get("profile", ["standard"])[0]
                days = int(qs.get("days", ["30"])[0])
                from instinct import decay_series, bump_series
                return self._send_json({
                    "score": score,
                    "profile": profile,
                    "days": days,
                    "decay": decay_series(score, profile, days=days, points=60),
                    "bump": bump_series(score, days=1, points=60),
                })
            if url.path == "/api/decay/preview":
                qs = parse_qs(url.query)
                from instinct import get_dormancy_config
                cfg = get_dormancy_config().to_dict()
                return self._send_json({"config": cfg, "scores": qs.get("scores", ["100,90,80,70,50,20"])[0].split(",")})
            return self._send_json(_err("not found", 404), 404)
        except RequestBoundaryError as error:
            return self._send_json(_err(str(error)), error.status)
        except Exception:
            sys.stderr.write('[dashboard] request failed; details withheld\n')
            return self._send_json(_err('operation failed; review configuration and retry'), 500)

    def do_POST(self):
        if not self._check_boundary(mutation=True):
            return
        url = urlparse(self.path)
        try:
            data = self._read_json()
            _validated_post_fields(data)
        except RequestBoundaryError as error:
            self.close_connection = True
            return self._send_json(_err(str(error)), error.status)

        try:


            if url.path.startswith("/api/mode/"):
                return self._send_json({"error": "SQLite service does not support writer handoff", "backend": "sqlite"}, 501)

            if url.path == "/api/ready":
                report = readiness_check()
                code = 200 if report["ok"] else 503
                return self._send_json(report, code)

            if url.path == "/api/admin/reset":
                include_recycle = data.get("include_recycle", False)
                dry_run = data.get("dry_run", False)
                if type(include_recycle) is not bool or type(dry_run) is not bool:
                    return self._send_json(_err("reset flags must be booleans"), 400)
                if not dry_run and is_read_only():
                    return self._send_json({"error": "reset requires write mode", "read_only": True}, 409)
                result = admin_reset(include_recycle=include_recycle, dry_run=dry_run)
                return self._send_json(result, result.get("code", 200))
            if url.path == "/api/heartbeat":


                return self._send_json({"error": "Heartbeat requires the SQLite service owner thread"}, 501)






            if is_read_only():
                _RO_ALLOW = {"/api/score/recommend", "/api/search_with_pop", "/api/ready"}
                if url.path not in _RO_ALLOW:
                    return self._send_json({
                        "error": "read-only SQLite observer: start an owner service after the observer stops.",
                        "code": 409,
                        "read_only": True,
                    }, 409)

            if url.path == "/api/search_with_pop":
                query = data.get("q", "")
                limit = data.get("n", 10)
                apply_pops = data.get("apply_pops", True)
                track_access = data.get("track_access", False)
                if not isinstance(query, str) or not query.strip():
                    return self._send_json(_err("nonempty q required"), 400)
                if type(limit) is not int or not 1 <= limit <= 100:
                    return self._send_json(_err("n must be an integer from 1 to 100"), 400)
                if type(apply_pops) is not bool or type(track_access) is not bool:
                    return self._send_json(_err("search flags must be booleans"), 400)
                if is_read_only() and (apply_pops or track_access):
                    return self._send_json({"error": "read-only search cannot mutate", "read_only": True}, 409)
                result = get_retriever().retrieve(
                    query, max_items=limit, apply_pops=apply_pops, track_access=track_access,
                )
                return self._send_json({
                    "query": query,
                    "applied_pops": apply_pops,
                    "tracked_access": track_access,
                    "mandatory_count": len(result.mandatory),
                    "relevant_count": len(result.relevant),
                    "pop_count": len(result.pop_results),
                    "mandatory": [_seg_to_dict(segment) for segment in result.mandatory],
                    "relevant": [_seg_to_dict(segment) for segment in result.relevant],
                    "pop_results": [vars(pop_result) for pop_result in result.pop_results],
                })



            if url.path == "/api/access":




                ids = data.get("ids") or []
                if not isinstance(ids, list) or not ids:
                    return self._send_json(_err("ids must be a non-empty list"), 400)


                try:
                    from instinct.nudge import tick_nudge_session
                    from instinct.dormancy import _cfg as _dorm_cfg
                    tick_nudge_session(get_store(), idle_minutes=float(_dorm_cfg().session_idle_minutes))
                except Exception:
                    sys.stderr.write('[dashboard] optional operation failed; details withheld\n')
                store = get_store()
                recorded = []
                for seg_id in ids:
                    if store.record_access(seg_id):
                        recorded.append(seg_id)
                return self._send_json({
                    "requested": len(ids),
                    "recorded": len(recorded),
                    "ids": recorded,
                })

            if url.path == "/api/score/recommend":
                cfg = _instinct_mod.get_dormancy_config()
                q1 = (data.get("q1") or "long").lower()
                q2 = (data.get("q2") or "solid").lower()
                rec = _recommend_score(q1, q2, cfg)
                return self._send_json({
                    "score": rec["score"],
                    "decay_profile": rec["decay_profile"],
                    "reasoning": rec["reasoning"],
                    "tier_label": rec["tier_label"],
                    "ephemeral": rec.get("ephemeral", False),
                    "scorecard": rec.get("scorecard", {}),
                })
            if url.path == "/api/score/ephemeral":
                cfg = _instinct_mod.get_dormancy_config()
                title = (data.get("title") or "").strip() or "Ephemeral note"
                content = (data.get("content") or "").strip()
                tags = data.get("tags") or []
                if isinstance(tags, str):
                    tags = [t.strip() for t in tags.split(",") if t.strip()]

                full_content = f"{title}\n\n{content}" if content else title
                seg = Segment(
                    content=full_content,
                    segment_type=SegmentType.CONTEXT,
                    metadata={"title": title, "tags": tags, "ephemeral": True},
                )
                scored = ScoredSegment(segment=seg, importance_score=80, concept_title=title,
                                        reasoning="ephemeral recall")
                _instinct_mod.start_ephemeral(scored, cfg)
                seg_id = get_store().add(scored)
                return self._send_json({"id": seg_id, "phase": scored.ephemeral_phase,
                                        "turns_left": scored.ephemeral_turns_left})
            if url.path == "/api/add":
                title, content, score, stype, profile, project, tags = _validated_segment_fields(data)
                meta = {"title": title, "tags": tags}
                if project:
                    meta["project"] = project
                seg = Segment(
                    content=content,
                    segment_type=stype,
                    metadata=meta,
                )
                scored = ScoredSegment(segment=seg, importance_score=score, concept_title=title,
                                        reasoning="dashboard add")

                scored.decay_profile = profile
                if profile == "ephemeral":
                    _instinct_mod.start_ephemeral(scored, _instinct_mod.get_dormancy_config())
                seg_id = get_store().add(scored)
                return self._send_json({
                    "id": seg_id, "score": scored.importance_score,
                    "decay_profile": scored.decay_profile,
                    "ephemeral_phase": getattr(scored, "ephemeral_phase", ""),
                })

            if url.path == "/api/segment/pop":
                seg_id = data.get("id")
                if not seg_id:
                    return self._send_json(_err("id required"), 400)
                matched = int(data.get("matched_keywords", 0))
                store = get_store()
                seg = store.get(seg_id)
                if not seg:
                    return self._send_json(_err("not found", 404), 404)
                from instinct import apply_pop
                cfg = _instinct_mod.get_dormancy_config()
                r = apply_pop(seg, matched, cfg)

                store.add(seg)
                return self._send_json({
                    "id": seg_id,
                    "popped": r.popped,
                    "rule_applied": r.rule_applied,
                    "matched_keywords": r.matched_keywords,
                    "initial_score_before": r.initial_score_before,
                    "initial_score_after": r.initial_score_after,
                    "effective_score_before": r.effective_score_before,
                    "effective_score_after": r.effective_score_after,
                    "anchor_reset": r.anchor_reset,
                    "message": r.message,
                })

            if url.path == "/api/nudge/run":


                from instinct import apply_nudge_check
                return self._send_json(apply_nudge_check(get_store()))

            if url.path == "/api/nudge/interval":


                seg_id = data.get("id")
                if not seg_id:
                    return self._send_json(_err("id required"), 400)
                if "turns" not in data:
                    return self._send_json(_err("turns required"), 400)
                turns = data["turns"]
                if type(turns) is not int or turns < 0:
                    return self._send_json(_err("turns must be >= 0"), 400)
                if "active" in data and type(data["active"]) is not bool:
                    return self._send_json(_err("active must be a boolean"), 400)
                store = get_store()
                seg = store.get(seg_id)
                if not seg:
                    return self._send_json(_err("not found", 404), 404)
                from instinct import set_nudge_interval, set_nudge_active, get_nudge_interval, nudge_active as _nudge_active
                set_nudge_interval(seg, turns)
                if "active" in data:
                    set_nudge_active(seg, bool(data["active"]))


                store.update_metadata_only(seg)
                return self._send_json({
                    "id": seg_id,
                    "nudge_interval_turns": get_nudge_interval(seg),
                    "nudge_active": _nudge_active(seg),
                })

            if url.path == "/api/recycle":



                seg_id = data.get("id")
                if not seg_id:
                    return self._send_json(_err("id required"), 400)
                reason = data.get("reason", "")
                by = data.get("by", "dashboard")
                seg = get_store().recycle(seg_id, by=by, reason=reason)
                if seg is None:
                    return self._send_json(_err("not found", 404), 404)
                return self._send_json({
                    "id": seg_id,
                    "recycled": True,
                    "recycled_at": (seg.segment.metadata or {}).get("recycled_at"),
                    "recycled_by": (seg.segment.metadata or {}).get("recycled_by"),
                    "recycled_reason": (seg.segment.metadata or {}).get("recycled_reason"),
                })

            if url.path == "/api/restore":
                seg_id = data.get("id")
                if not seg_id:
                    return self._send_json(_err("id required"), 400)
                seg = get_store().restore(seg_id)
                if seg is None:
                    return self._send_json(_err("not found", 404), 404)
                return self._send_json({"id": seg_id, "restored": True})

            if url.path == "/api/purge":




                seg_id = data.get("id")
                if not seg_id:
                    return self._send_json(_err("id required"), 400)
                ok = get_store().delete(seg_id)
                return self._send_json({"id": seg_id, "purged": ok})

            if url.path == "/api/delete":



                seg_id = data.get("id")
                if not seg_id:
                    return self._send_json(_err("id required"), 400)
                reason = data.get("reason", "legacy /api/delete")
                seg = get_store().recycle(seg_id, by="dashboard-legacy", reason=reason)
                if seg is None:
                    return self._send_json(_err("not found", 404), 404)
                return self._send_json({
                    "id": seg_id,
                    "recycled": True,
                    "deprecated": "use /api/recycle or /api/purge",
                })

            if url.path == "/api/update":
                import copy
                from instinct.models import build_memory_identifiers, MANDATORY_TYPES

                seg_id = data.get("id")
                if not isinstance(seg_id, str) or not seg_id:
                    raise ValueError("id required")
                title, content, score, stype, profile, project, tags = _validated_segment_fields(data)
                store = get_store()
                existing = store.get(seg_id)
                if existing is None:
                    return self._send_json(_err("not found", 404), 404)
                seg = copy.deepcopy(existing)
                seg.segment.stored_id = seg_id
                seg.segment.content = content
                seg.segment.segment_type = stype
                seg.segment.metadata.update({"title": title, "project": project, "tags": tags})
                seg.concept_title = title
                seg.memory_identifiers = build_memory_identifiers(title, content)
                seg.importance_score = 100 if stype in MANDATORY_TYPES else score
                seg.decay_profile = profile
                if seg.importance_score != existing.importance_score or profile != existing.decay_profile:
                    from datetime import datetime
                    from .active_time import set_anchor
                    seg.pop_anchor_at = datetime.now()
                    set_anchor(seg, "pop_anchor_at", store.persist_dir)
                if profile == "ephemeral" and existing.decay_profile != "ephemeral":
                    _instinct_mod.start_ephemeral(seg, _instinct_mod.get_dormancy_config())
                saved_id = store.add(seg)
                if saved_id != seg_id:
                    raise RuntimeError("update was not confirmed")
                return self._send_json({"id": saved_id, "updated": True})

            if url.path == "/api/update_score":
                seg_id = data.get("id")
                if not seg_id:
                    return self._send_json(_err("id required"), 400)
                store = get_store()
                seg = store.get(seg_id)
                if not seg:
                    return self._send_json(_err("not found", 404), 404)
                score = data.get("score")
                if score is not None:
                    if type(score) is not int or score < 0 or score > 100:
                        return self._send_json(_err("score must be 0..100"), 400)
                    seg.importance_score = score
                reasoning = data.get("reasoning", "dashboard update")
                if reasoning and hasattr(seg, "reasoning"):
                    seg.reasoning = reasoning

                if "decay_profile" in data:
                    profile = data["decay_profile"]
                    if profile not in ("standard", "short_term_critical"):
                        return self._send_json(_err("invalid decay_profile"), 400)
                    seg.decay_profile = profile

                store.add(seg)
                return self._send_json({
                    "id": seg_id,
                    "score": seg.importance_score,
                    "decay_profile": seg.decay_profile,
                    "updated": True,
                })


            if url.path == "/api/decay/config":
                from instinct import update_dormancy_config
                from instinct import get_dormancy_config
                from instinct.dormancy import DormancyConfig
                current = get_dormancy_config()
                updates = _validated_config_updates(data, current)
                candidate = DormancyConfig.from_dict({**current.to_dict(), **updates})
                cfg = update_dormancy_config(**{key: getattr(candidate, key) for key in updates})
                return self._send_json({"ok": True, "config": cfg.to_dict()})

            if url.path == "/api/decay/tick":



                try:
                    from instinct.nudge import tick_nudge_session
                    from instinct.dormancy import _cfg as _dorm_cfg
                    tick_nudge_session(get_store(), idle_minutes=float(_dorm_cfg().session_idle_minutes))
                except Exception:
                    sys.stderr.write('[dashboard] optional operation failed; details withheld\n')
                from datetime import datetime
                from instinct import (
                    should_bump_to_surface,
                    current_effective_score,
                )
                store = get_store()
                now = datetime.now()
                all_segs = store.get_by_score(min_score=0, max_score=100, limit=10000)
                bumped = decayed = 0
                from instinct.models import MANDATORY_TYPES
                for seg in all_segs:
                    original = seg.importance_score
                    if original >= 99 and should_bump_to_surface(seg, now):
                        seg.last_bumped_at = now
                        seg.pop_anchor_at = now
                        from .active_time import set_anchor
                        set_anchor(seg, "last_bumped_at", store.persist_dir)
                        set_anchor(seg, "pop_anchor_at", store.persist_dir)
                        seg.bump_count = (seg.bump_count or 0) + 1



                        store.update_metadata_only(seg)
                        bumped += 1






                    if seg.segment.segment_type in MANDATORY_TYPES:
                        continue
                    new_score = int(round(current_effective_score(seg, now)))
                    if new_score != original:
                        seg.importance_score = new_score



                        store.update_metadata_only(seg)
                        decayed += 1
                return self._send_json({
                    "ts": now.isoformat(), "scanned": len(all_segs),
                    "bumped": bumped, "decayed": decayed,
                })

            return self._send_json(_err("not found", 404), 404)
        except RequestBoundaryError as error:
            return self._send_json(_err(str(error)), error.status)
        except (ValueError, TypeError):
            return self._send_json(_err('invalid operation parameters; details withheld'), 400)
        except Exception:
            sys.stderr.write('[dashboard] request failed; details withheld\n')
            return self._send_json(_err('operation failed; review configuration and retry'), 500)









MODE_BANNER_HTML = r"""
<style>
  .mode-banner { display: flex; align-items: center; gap: 8px;
                 flex-wrap: wrap;
                 padding: 6px 12px; border-radius: 4px;
                 font-size: 12px; font-weight: 600;
                 margin: 8px 12px 0; box-shadow: 0 2px 8px rgba(0,0,0,.4); }
  .mode-banner.write    { background: var(--success); border: 1px solid var(--success-h); color: #fff; }
  .mode-banner.read_only{ background: var(--danger); border: 1px solid var(--danger-h); color: #fff; }
  .mode-banner .status { font-size: 11px; color: inherit; }
  .mode-banner .read-only-preference { display: inline-flex; align-items: center; gap: 5px; margin-left: auto; font-weight: 500; cursor: pointer; }
  .mode-banner .read-only-preference-note { font-size: 10px; font-weight: 400; color: inherit; opacity: .9; }
  .mode-banner .handoff-actions { display: flex; align-items: center; gap: 6px; }
  .mode-banner button[hidden] { display: none !important; }
  header:not(.app-bar) > a[href="/"],
  header:not(.app-bar) > a[href="/decay"],
  header:not(.app-bar) > a[href="/score"] {
    display: inline-flex; align-items: center; min-height: 38px;
    margin: 4px 6px 4px 0; padding: 0 12px;
    border: 1px solid var(--border); border-radius: 8px;
    background: var(--surface-2); color: var(--text);
    font-size: 13px; font-weight: 550; text-decoration: none;
  }
  header:not(.app-bar) > a[href="/"]:hover,
  header:not(.app-bar) > a[href="/decay"]:hover,
  header:not(.app-bar) > a[href="/score"]:hover {
    background: var(--accent); border-color: var(--accent); color: #fff;
  }
</style>
<div class="mode-banner" id="mode-banner">
  <span class="status" id="mode-banner-status">checking mode...</span>
  <span>One SQLite owner serves MCP and dashboard requests.</span>
  <span class="handoff-actions" aria-label="Writer handoff controls">
    <button type="button" id="handoff-to-dashboard" hidden>Take Dashboard Control</button>
    <button type="button" id="handoff-to-mcp" hidden>Return to MCP</button>
  </span>
  <label class="read-only-preference" title="Browser-local preference; the SQLite service still enforces write permissions.">
    <input type="checkbox" id="instinct-prefer-readonly" aria-describedby="read-only-preference-note">
    <span>Prefer read-only</span>
  </label>
  <span class="read-only-preference-note" id="read-only-preference-note">Dashboard-local preference only.</span>
</div>

<script>
(async function modeInit() {
  const banner = document.getElementById('mode-banner');
  const status = document.getElementById('mode-banner-status');
  const takeButton = document.getElementById('handoff-to-dashboard');
  const returnButton = document.getElementById('handoff-to-mcp');
  // Keep the owner SQLite session active while the dashboard is in use.
  let heartbeatTimer = null;
  let inWriteMode = false;
  let handoffBusy = false;

  function startHeartbeat() {
    if (heartbeatTimer) return;
    inWriteMode = true;
    // Send immediately, then every 10s
    sendHeartbeat();
    heartbeatTimer = setInterval(sendHeartbeat, 10000);
  }
  function stopHeartbeat() {
    inWriteMode = false;
    if (heartbeatTimer) { clearInterval(heartbeatTimer); heartbeatTimer = null; }
  }
  async function sendHeartbeat() {
    try {
      await dashboardFetch('/api/heartbeat', { method: 'POST', keepalive: true });
    } catch (e) { /* ignore — watchdog will handle timeout */ }
  }

  async function handoff(path, button, label) {
    if (handoffBusy) return;
    handoffBusy = true;
    button.disabled = true;
    button.textContent = 'Transferring...';
    let handoffError = '';
    try {
      const response = await dashboardFetch(path, { method: 'POST', body: '{}' });
      const result = await response.json();
      if (!response.ok) throw new Error(result.message || result.error || 'Handoff was refused');
      status.textContent = result.message || 'Write control transferred.';
      await refresh();
    } catch (error) {
      handoffError = label + ' failed: ' + error.message;
    } finally {
      handoffBusy = false;
      button.textContent = label;
      await refresh();
      if (handoffError) status.textContent = handoffError;
    }
  }

  takeButton.addEventListener('click', () =>
    handoff('/api/mode/take_control', takeButton, 'Take Dashboard Control'));
  returnButton.addEventListener('click', () =>
    handoff('/api/mode/return_control', returnButton, 'Return to MCP'));

  // Refresh mode and start/stop heartbeat accordingly
  async function refresh() {
    try {
      const r = await dashboardFetch('/api/mode');
      const j = await r.json();
      if (j.backend === 'sqlite') {
        const dashboardOwnsControl = j.dashboard_control === true;
        const mcpAvailable = j.mcp_available === true;
        const handoffAvailable = j.handoff_supported === true;
        instinctReadOnly = !dashboardOwnsControl;
        banner.className = 'mode-banner ' + (dashboardOwnsControl ? 'write' : 'read_only');
        status.textContent = dashboardOwnsControl
          ? 'Dashboard has write control; MCP tools are paused.'
          : mcpAvailable && handoffAvailable
            ? 'MCP is connected; dashboard write control is available.'
            : mcpAvailable
              ? 'MCP is connected; dashboard handoff is unavailable with queue capture enabled.'
            : 'SQLite observer (browse only; connect MCP to hand off write control).';
        takeButton.hidden = !mcpAvailable || dashboardOwnsControl || !handoffAvailable;
        returnButton.hidden = !mcpAvailable || !dashboardOwnsControl;
        takeButton.disabled = handoffBusy || !handoffAvailable;
        returnButton.disabled = handoffBusy;
        dashboardApplyWriteState();
        if (!dashboardOwnsControl) stopHeartbeat(); else startHeartbeat();
        return;
      }
      status.textContent = 'Unsupported storage service';
      stopHeartbeat();
    } catch (e) {
      status.textContent = 'mode check failed: ' + e;
    }
  }

  window.addEventListener('beforeunload', function() {
    if (inWriteMode) {
      dashboardFetch('/api/heartbeat', { method: 'POST', keepalive: true }).catch(() => {});
    }
  });

  // Re-check service mode periodically.
  setInterval(refresh, 15000);

  await refresh();
})();
</script>
"""


MODIFIER_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Vecretary Memory - Modifier Window</title>
<style>
  :root { color-scheme: light dark; }
  body { font: 14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif;
         margin: 0; background: #111; color: #eee; }
  header { background: #1d1d1d; padding: 12px 20px; border-bottom: 1px solid #333;
           display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 16px; margin: 0; font-weight: 600; }
  header a { color: var(--brand-bright); text-decoration: none; }
  header .meta { font-size: 11px; color: #888; }
  main { max-width: 920px; margin: 0 auto; padding: 16px; }
  .picker { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 6px;
            padding: 10px 12px; margin-bottom: 14px; display: flex; gap: 8px; align-items: center; }
  .picker select { background: #222; color: #eee; border: 1px solid #333;
                   border-radius: 4px; padding: 5px 8px; font: inherit; min-width: 380px; }
  .picker input { background: #222; color: #eee; border: 1px solid #333;
                  border-radius: 4px; padding: 5px 8px; font: inherit; width: 380px; }
  .picker button { background: var(--accent); color: #fff; border: 0; border-radius: 4px;
                   padding: 6px 12px; cursor: pointer; font: inherit; }
  .picker button:hover { background: var(--accent-h); }
  .panel { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 6px;
           padding: 12px; margin-bottom: 14px; }
  .panel h2 { font-size: 13px; margin: 0 0 8px 0; text-transform: uppercase;
              letter-spacing: .05em; color: #999; }
  .seg-title { font-size: 16px; font-weight: 600; margin-bottom: 4px; }
  .seg-meta { font-size: 11px; color: #888; }
  /* modifier window pipeline */
  .pipeline { display: flex; flex-direction: column; gap: 8px; }
  .stage { display: grid; grid-template-columns: 160px 60px 1fr 110px;
           gap: 8px; align-items: center; background: #161616;
           border: 1px solid #2a2a2a; border-radius: 6px; padding: 10px 12px; }
  .stage.initial { border-color: var(--brand-deep); }
  .stage.final   { border-color: var(--brand-bright); }
  .stage .label  { font-weight: 600; font-size: 13px; }
  .stage .label .num { display: block; font-size: 11px; color: #888;
                       font-weight: 400; margin-top: 2px; }
  .stage .gross  { text-align: center; font-size: 18px; font-weight: 700;
                   padding: 4px 8px; border-radius: 4px; background: #222; }
  .stage .gross.neg { color: #fa6; }
  .stage .gross.pos { color: var(--brand-bright); }
  .stage .gross.zero { color: #888; }
  .stage .desc   { font-size: 12px; color: #aaa; line-height: 1.5; }
  .stage .desc code { background: #222; padding: 1px 5px; border-radius: 2px; font-size: 11px; }
  .stage .desc .boxes { display: grid; grid-template-columns: repeat(4, 1fr);
                        gap: 4px; margin-top: 6px; }
  .stage .desc .box { background: #1a1a1a; border: 1px solid #2a2a2a;
                      border-radius: 4px; padding: 4px 6px; text-align: center; }
  .stage .desc .box .b { font-weight: 700; color: #fff; font-size: 14px; }
  .stage .desc .box .l { font-size: 10px; color: #888; text-transform: uppercase; }
  .stage .desc .flags { margin-top: 6px; font-size: 11px; }
  .stage .desc .flags .on  { color: var(--brand-bright); }
  .stage .desc .flags .off { color: #666; }
  .stage .result { text-align: center; font-size: 22px; font-weight: 700;
                   padding: 6px 10px; border-radius: 4px; background: #222; }
  .arrow { text-align: center; color: #555; font-size: 14px; margin: -2px 0; }
  /* toolbar */
  .actions { display: flex; gap: 8px; margin-top: 6px; }
  .actions button { background: var(--accent); color: #fff; border: 0; border-radius: 4px;
                    padding: 6px 12px; cursor: pointer; font: inherit; }
  .actions button:hover { background: var(--accent-h); }
  .actions button.pop { background: var(--brand-deep); }
  .actions button.pop:hover { background: var(--accent-h); }
  .toast { position: fixed; bottom: 16px; right: 16px; background: var(--brand-deep);
           padding: 8px 12px; border-radius: 4px; display: none; z-index: 50; }
  .toast.error { background: #7a2a2a; }
  .error-msg { color: #fa6; font-size: 13px; padding: 12px; }
  .info { font-size: 12px; color: #888; }
</style>
</head>
<body>
<header>
  <h1>Vecretary Memory &mdash; Modifier Window</h1>
  <a href="/">&laquo; back to segments</a>
  <a href="/decay">/decay</a>
  <a href="/score">/score</a>
  <div style="flex:1"></div>
  <span class="meta" id="meta"></span>
</header>
<main>
  <div class="picker">
    <span>Segment:</span>
    <select id="seg-select"><option value="">loading...</option></select>
    <span>or</span>
    <input id="seg-id" placeholder="paste segment id">
    <button onclick="loadFromInput()">Load</button>
    <button onclick="reload()" title="Re-read modifier window">Refresh</button>
  </div>

  <div class="panel">
    <h2>Modifier pipeline</h2>
    <p class="info">
      Each section shows one modifier's <b>gross effect</b> (rounded integer, signed)
      and the running <b>resulting</b> score after that step. The last section is
      the current modified score.
    </p>
    <div class="pipeline" id="pipeline"></div>
    <div class="actions">
      <button class="pop" onclick="applyPop()">Apply pop (simulate 2 keyword hits)</button>
      <button onclick="simulateBump()" id="bump-btn">Simulate bump (timestamp = now)</button>
    </div>
  </div>
</main>
<div class="toast" id="toast"></div>
<script>
const $ = id => document.getElementById(id);
let currentId = null;
let segments = [];

function toast(msg, isError) {
  const t = $('toast');
  t.textContent = msg;
  t.className = 'toast' + (isError ? ' error' : '');
  t.style.display = 'block';
  setTimeout(() => t.style.display = 'none', 2500);
}

async function api(path, opts = {}) {
  const r = await dashboardFetch(path, {
    method: opts.method || 'GET',
    headers: opts.body ? {'Content-Type': 'application/json'} : {},
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || ('HTTP ' + r.status));
  return j;
}

function fmtGross(g) {
  if (g === 0) return { text: '0', cls: 'zero' };
  const sign = g > 0 ? '+' : '';
  return { text: sign + g, cls: g > 0 ? 'pos' : 'neg' };
}

function renderInitial(stage) {
  return `
    <div class="stage initial">
      <div class="label">Initial Score<span class="num">raw starting value</span></div>
      <div></div>
      <div class="desc">Score when this segment was first saved. Decay modifiers
        subtract from (or pop back toward) this value.</div>
      <div class="result">${stage.initial_score}</div>
    </div>`;
}

function renderSection(stage, isFinal) {
  const g = fmtGross(stage.gross_effect);
  let extra = '';
  if (stage.boxes) {
    const b = stage.boxes;
    extra += `<div class="boxes">
      <div class="box"><div class="b">${b.hits_8h_active ?? 0}</div><div class="l">8h active</div></div>
      <div class="box"><div class="b">${b.hits_16h_active ?? 0}</div><div class="l">16h active</div></div>
      <div class="box"><div class="b">${b.hits_24h_active ?? 0}</div><div class="l">24h active</div></div>
      <div class="box"><div class="b">${b.hits_total ?? 0}</div><div class="l">total</div></div>
    </div>`;
    if (stage.flags) {
      const h = stage.flags.halve_applied ? '<span class="on">[x] 1/2 timer applied</span>' : '<span class="off">[ ] 1/2 timer applied</span>';
      const r = stage.flags.reset_applied ? '<span class="on">[x] reset timer applied</span>' : '<span class="off">[ ] reset timer applied</span>';
      extra += `<div class="flags">${h} &nbsp; ${r}</div>`;
    }
  }
  return `
    <div class="arrow">&#x25BC;</div>
    <div class="stage ${isFinal ? 'final' : ''}">
      <div class="label">${dashboardEscapeHtml(stage.name)}${stage.config && stage.config.band ? ' <span class="num">band='+dashboardEscapeHtml(stage.config.band)+'</span>' : ''}</div>
      <div class="gross ${g.cls}" title="gross effect (rounded integer)">${g.text}</div>
      <div class="desc">${dashboardEscapeHtml(stage.description)}${extra}</div>
      <div class="result">${stage.resulting}</div>
    </div>`;
}

function renderWindow(data) {
  const m = data.modifier_window;
  const pipe = $('pipeline');
  let html = renderInitial({ initial_score: m.initial_score });
  m.sections.forEach((s, i) => {
    html += renderSection(s, i === m.sections.length - 1);
  });
  pipe.innerHTML = html;
  $('meta').textContent =
    `id ${data.id.slice(0,8)} &middot; band=${m.band} &middot; profile=${m.decay_profile} &middot; final=${m.final_modified_score}`;
}

async function loadId(id) {
  if (!id) return;
  currentId = id;
  try {
    const r = await api('/api/segment/modifier_window?id=' + encodeURIComponent(id));
    renderWindow(r);
    // Update URL without reload
    const u = new URL(window.location.href);
    u.searchParams.set('id', id);
    window.history.replaceState({}, '', u);
  } catch (e) {
    $('pipeline').textContent = 'Error: ' + e.message;
  }
}

function loadFromInput() {
  const id = $('seg-id').value.trim();
  if (id) loadId(id);
}

async function reload() {
  if (currentId) loadId(currentId);
}

async function populateDropdown() {
  try {
    const r = await api('/api/list?limit=200');
    segments = r.items;
    const sel = $('seg-select');
    sel.innerHTML = '<option value="">— choose segment —</option>' +
      segments.map(s => `<option value="${dashboardEscapeHtml(s.id)}">${s.score || 0} | ${dashboardEscapeHtml((s.title || s.content.slice(0, 60)).slice(0, 80))}</option>`).join('');
    sel.onchange = e => { if (e.target.value) loadId(e.target.value); };
    // Auto-load from ?id=
    const url = new URL(window.location.href);
    const idParam = url.searchParams.get('id');
    if (idParam) {
      $('seg-id').value = idParam;
      loadId(idParam);
      const opt = Array.from(sel.options).find(o => o.value === idParam);
      if (opt) sel.value = idParam;
    }
  } catch (e) { toast(String(e), true); }
}

async function applyPop() {
  if (!currentId) { toast('Load a segment first', true); return; }
  try {
    await api('/api/segment/pop', {
      method: 'POST',
      body: { id: currentId, matched_keywords: 2 },
    });
    toast('Pop applied (2 keywords)');
    await loadId(currentId);
  } catch (e) { toast(String(e), true); }
}

async function simulateBump() {
  if (!currentId) { toast('Load a segment first', true); return; }
  // We piggy-back on the record_access endpoint: it stamps last_bumped_at
  // for mandatory segments. For non-mandatory it still updates access log.
  try {
    await api('/api/access', { method: 'POST', body: { ids: [currentId] } });
    toast('Access recorded (last_bumped_at updated if mandatory)');
    await loadId(currentId);
  } catch (e) { toast(String(e), true); }
}

populateDropdown();
</script>
</body>
</html>
"""

DECAY_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Vecretary Memory - Dormancy & Decay</title>
<style>
  :root { color-scheme: light dark; }
  body { font: 14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif;
         margin: 0; background: #111; color: #eee; }
  header { background: #1d1d1d; padding: 12px 20px; border-bottom: 1px solid #333;
           display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 16px; margin: 0; font-weight: 600; }
  header a { color: var(--brand-bright); text-decoration: none; }
  main { display: grid; grid-template-columns: 1fr 360px; gap: 16px; padding: 16px; }
  .panel { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 6px; padding: 12px; }
  .panel h2 { font-size: 13px; margin: 0 0 8px 0; text-transform: uppercase; letter-spacing: .05em; color: #999; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 12px; margin-bottom: 8px; }
  .row.three { grid-template-columns: 1fr 1fr 1fr; }
  label { display: block; font-size: 11px; color: #999; margin-bottom: 2px; }
  input, select, button { background: #222; color: #eee; border: 1px solid #333;
                          border-radius: 4px; padding: 5px 8px; font: inherit; box-sizing: border-box; }
  input[type=number] { width: 100%; }
  button { background: var(--accent); cursor: pointer; }
  button:hover { background: var(--accent-h); }
  button.tiny { padding: 2px 6px; font-size: 11px; }
  .toast { position: fixed; bottom: 16px; right: 16px; background: var(--brand-deep);
           padding: 8px 12px; border-radius: 4px; display: none; }
  .toast.error { background: #7a2a2a; }
  svg { background: #161616; border-radius: 4px; display: block; width: 100%; height: auto; }
  .legend { font-size: 11px; color: #888; margin-top: 6px; line-height: 1.6; }
  .legend span.swatch { display: inline-block; width: 10px; height: 10px;
                        vertical-align: middle; margin-right: 4px; border-radius: 2px; }
  .stat { font-size: 11px; color: #aaa; }
  .stat b { color: #fff; }
  details { margin-top: 6px; }
  summary { cursor: pointer; color: var(--brand-bright); font-size: 12px; }
</style>
</head>
<body>
<header>
  <h1>Vecretary Memory &mdash; Dormancy &amp; Decay</h1>
  <a href="/">&laquo; back to segments</a>
  <div style="flex:1"></div>
  <span class="stat" id="stat">loading config...</span>
  <button onclick="runTick()">Run tick now</button>
</header>
<main>
  <section class="panel">
    <h2>Decay projection (continuous active time)</h2>
    <p id="selected-preview" class="stat">Hypothetical continuous activity, not a stored history.</p>
    <svg id="graph" viewBox="0 0 800 400" preserveAspectRatio="none"></svg>
    <div class="legend" id="legend"></div>
    <details>
      <summary>Show bump sawtooth for score 100 (1-day window)</summary>
      <svg id="bumpGraph" viewBox="0 0 800 200" preserveAspectRatio="none"></svg>
    </details>
  </section>
  <aside class="panel">
    <h2>Tunables</h2>
    <p class="stat">All values editable. Edits persist to <code>dormancy_config.json</code>.</p>

    <div id="tunables"></div>

    <div style="display:flex;gap:6px;margin-top:8px">
      <button onclick="saveAll()">Save &amp; redraw</button>
      <button onclick="reload()">Reload</button>
    </div>
    <p class="stat" id="last-saved"></p>
  </aside>
</main>
<div class="toast" id="toast"></div>
<script>
let config = {};
const SCORES = [100, 90, 80, 70, 50, 20];
const SCORE_LINE_STYLES = {
  100: {color: 'var(--decay-score-100)', dash: ''},
  90: {color: 'var(--decay-score-90)', dash: '6 2'},
  80: {color: 'var(--decay-score-80)', dash: '2 2'},
  70: {color: 'var(--decay-score-70)', dash: '8 3'},
  50: {color: 'var(--decay-score-50)', dash: '4 2'},
  20: {color: 'var(--decay-score-20)', dash: '1 3'},
};

function lineStyleForScore(score) {
  if (score === 100) return SCORE_LINE_STYLES[100];
  if ([90, 94, 95].includes(score)) return SCORE_LINE_STYLES[90];
  if (score >= 80) return SCORE_LINE_STYLES[80];
  if (score >= 65) return SCORE_LINE_STYLES[70];
  if (score >= 50) return SCORE_LINE_STYLES[50];
  return SCORE_LINE_STYLES[20];
}

function toast(msg, isError) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'toast' + (isError ? ' error' : '');
  t.style.display = 'block';
  setTimeout(() => t.style.display = 'none', 2200);
}

async function api(path, opts = {}) {
  const r = await dashboardFetch(path, {
    method: opts.method || 'GET',
    headers: opts.body ? {'Content-Type': 'application/json'} : {},
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || ('HTTP ' + r.status));
  return j;
}

function fillForm() {
  const root = document.querySelector('aside.panel');
  root.querySelectorAll('.row').forEach(row => row.remove());
  let fields = document.getElementById('tunables');
  if (!fields) { fields = document.createElement('div'); fields.id = 'tunables'; root.insertBefore(fields, root.querySelector('div[style]')); }
  fields.replaceChildren();
  const tierGroups = new Map();
  function addFields(values, parent, prefix = '') {
    for (const [key, value] of Object.entries(values)) {
      if (key === 'session_state') continue;
      const path = prefix + key;
      if (value && typeof value === 'object') {
        const group = document.createElement('fieldset');
        group.dataset.range = key;
        const legend = document.createElement('legend'); legend.textContent = ({plateau: 'plateau (80-89; upper standard fallback)', mid: 'mid (50-79)', tail: 'tail (0-49)', mandatory: 'mandatory (99-100)'})[key] || key.replaceAll('_', ' '); group.appendChild(legend);
        addFields(value, group, path + '.'); parent.appendChild(group); continue;
      }
      const label = document.createElement('label'); label.textContent = key.replaceAll('_', ' ');
      let fieldParent = parent;
      const tierMatch = !prefix && key.match(/^tier_(90|94|95)_/);
      if (tierMatch) {
        const tier = tierMatch[1];
        fieldParent = tierGroups.get(tier);
        if (!fieldParent) {
          fieldParent = document.createElement('fieldset');
          fieldParent.dataset.range = 'tier-' + tier;
          const legend = document.createElement('legend'); legend.textContent = 'tier ' + tier;
          fieldParent.appendChild(legend);
          tierGroups.set(tier, fieldParent);
          parent.appendChild(fieldParent);
        }
      }
      let control;
      if (key === 'decay_function') {
        control = document.createElement('select');
        for (const option of ['linear','exponential','half_life']) { const item = document.createElement('option'); item.value = option; item.textContent = option; control.appendChild(item); }
      } else {
        control = document.createElement('input'); control.type = 'number'; control.min = '0'; control.step = 'any';
      }
      control.dataset.field = path; control.value = value; label.appendChild(control); fieldParent.appendChild(label);
    }
  }
  addFields(config, fields);
}

function readForm() {
  const out = {};
  document.querySelectorAll('[data-field]').forEach(el => {
    const value = el.tagName === 'SELECT' ? el.value : Number(el.value);
    if (el.tagName !== 'SELECT' && (!el.value.trim() || !Number.isFinite(value) || value < 0)) throw new Error('Invalid configuration value: ' + el.dataset.field);
    const keys = el.dataset.field.split('.');
    let target = out;
    for (const key of keys.slice(0,-1)) target = target[key] ||= {};
    target[keys.at(-1)] = value;
  });
  return out;
}

function drawGraph(series) {
  const W = 800, H = 400, P = 30;
  const maxMin = Math.max(...series.flatMap(s => s.decay.map(p => p.t_min)));
  const yMax = 100;
  const x = m => P + (m / maxMin) * (W - 2*P);
  const y = s => H - P - (s / yMax) * (H - 2*P);
  let svg = '';
  // gridlines
  for (let g = 0; g <= 100; g += 20) {
    svg += `<line x1="${P}" x2="${W-P}" y1="${y(g)}" y2="${y(g)}" stroke="#333" stroke-width="0.5"/>`;
    svg += `<text x="${P-4}" y="${y(g)+3}" fill="#888" font-size="10" text-anchor="end">${g}</text>`;
  }
  // x labels
  for (let d = 0; d <= 30; d += 5) {
    const min = d * 1440;
    svg += `<line x1="${x(min)}" x2="${x(min)}" y1="${P}" y2="${H-P}" stroke="#333" stroke-width="0.5"/>`;
    svg += `<text x="${x(min)}" y="${H-P+12}" fill="#888" font-size="10" text-anchor="middle">${d}d</text>`;
  }
  // series
  series.forEach(s => {
    const pts = s.decay.map(p => `${x(p.t_min)},${y(p.score)}`).join(' ');
    const style = lineStyleForScore(s.score);
    const dash = s.selected ? '6 4' : style.dash;
    const dashAttr = dash ? ` stroke-dasharray="${dash}"` : '';
    svg += `<polyline points="${pts}" fill="none" stroke="${style.color}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" opacity="0.95"${dashAttr}/>`;
  });
  // dormant floor line
  svg += `<line x1="${P}" x2="${W-P}" y1="${y(config.mandatory.floor)}" y2="${y(config.mandatory.floor)}" stroke="var(--decay-floor-color)" stroke-dasharray="3,3" stroke-width="1"/>`;
  svg += `<text x="${W-P-4}" y="${y(config.mandatory.floor)-4}" fill="var(--decay-floor-color)" font-size="10" text-anchor="end">mandatory floor (${config.mandatory.floor})</text>`;
  document.getElementById('graph').innerHTML = svg;
  document.getElementById('legend').innerHTML = series.map(s => {
    const style = lineStyleForScore(s.score);
    return `<span><span class="swatch" style="background:${style.color}"></span>${s.selected ? 'selected · ' : ''}score ${s.score}</span>`;
  }).join(' &nbsp; ') + ` &nbsp; <span><span class="swatch" style="background:var(--decay-floor-color)"></span>mandatory floor</span>`;
}

function drawBump(series) {
  const W = 800, H = 200, P = 20;
  const pts = series.bump;
  if (!pts.length) { document.getElementById('bumpGraph').innerHTML = ''; return; }
  const maxMin = pts[pts.length-1].t_min || 1;
  const x = m => P + (m / maxMin) * (W - 2*P);
  const y = s => H - P - (s / 100) * (H - 2*P);
  let svg = '';
  for (let g = 0; g <= 100; g += 25) {
    svg += `<line x1="${P}" x2="${W-P}" y1="${y(g)}" y2="${y(g)}" stroke="#333" stroke-width="0.5"/>`;
  }
  const path = pts.map(p => `${x(p.t_min)},${y(p.score)}`).join(' ');
  svg += `<polyline points="${path}" fill="none" stroke="${lineStyleForScore(100).color}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>`;
  document.getElementById('bumpGraph').innerHTML = svg;
}

async function reload() {
  try {
    config = await api('/api/decay/config');
    fillForm();
    await redraw();
    document.getElementById('stat').textContent =
      `mandatory floor=${config.mandatory.floor} · bump base=${config.bump_base_active_hours} active hours · tail half-life=${config.tail.post_drop_half_life_active_hours} active hours`;
  } catch (e) { toast(String(e), true); }
}

async function redraw() {
  try {
    const series = [];
    const selectedId = new URL(location.href).searchParams.get('id');
    if (selectedId) {
      const selected = await api('/api/get?id=' + encodeURIComponent(selectedId));
      document.getElementById('selected-preview').textContent = 'Projection for ' + selected.title + ' (continuous active time, not stored history)';
      const selectedSeries = await api('/api/decay/series?score=' + selected.score + '&profile=' + encodeURIComponent(selected.decay_profile) + '&days=30');
      selectedSeries.selected = true;
      series.push(selectedSeries);
    }
    for (const score of SCORES) {
      const s = await api('/api/decay/series?score=' + score + '&profile=standard&days=30');
      series.push(s);
    }
    drawGraph(series);
    if (series.length) drawBump(series.find(item => item.score === 100));
  } catch (e) { toast(String(e), true); }
}

async function saveAll() {
  try {
    const updates = readForm();
    const r = await api('/api/decay/config', { method: 'POST', body: updates });
    config = r.config;
    fillForm();
    document.getElementById('last-saved').textContent = 'saved at ' + new Date().toLocaleTimeString();
    await redraw();
    toast('Saved');
  } catch (e) { toast(String(e), true); }
}

async function runTick() {
  try {
    const r = await api('/api/decay/tick', {method:'POST', body:{}});
    toast('tick: scanned=' + r.scanned + ' bumped=' + r.bumped + ' decayed=' + r.decayed);
  } catch (e) { toast(String(e), true); }
}

(async function init() { await reload(); })();
</script>
</body>
</html>
"""


SCORE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Vecretary Memory - Score Helper</title>
<style>
  :root { color-scheme: light dark; }
  body { font: 14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif;
         margin: 0; background: #111; color: #eee; }
  header { background: #1d1d1d; padding: 12px 20px; border-bottom: 1px solid #333;
           display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 16px; margin: 0; font-weight: 600; }
  header a { color: var(--brand-bright); text-decoration: none; }
  main { max-width: 820px; margin: 0 auto; padding: 16px; }
  .panel { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 6px; padding: 16px; margin-bottom: 14px; }
  .panel h2 { font-size: 13px; margin: 0 0 12px 0; text-transform: uppercase; letter-spacing: .05em; color: #999; }
  .q { display: block; padding: 8px 12px; margin-bottom: 6px; background: #222; border: 1px solid #333;
       border-radius: 4px; cursor: pointer; }
  .q:hover { background: #2a2a2a; }
  .q input { margin-right: 8px; }
  .q.selected { background: #172d1f; border-color: #5bec7d; }
  .opt { display: block; padding: 6px 10px; margin: 4px 0 4px 24px; background: #1a1a1a;
         border: 1px solid #2a2a2a; border-radius: 4px; cursor: pointer; }
  .opt:hover { background: #222; }
  .opt.selected { background: #172d1f; border-color: #5bec7d; }
  .opt small { color: #999; display: block; margin-top: 2px; }
  .recommend { background: #172d1f; border-color: #246f3a; padding: 14px; border-radius: 4px; display: none; }
  .recommend.show { display: block; }
  .recommend h3 { margin: 0 0 6px 0; color: var(--brand-bright); font-size: 14px; }
  .recommend .score { font-size: 24px; font-weight: 700; color: #b9f5c7; margin: 6px 0; }
  .recommend .reasoning { font-size: 12px; color: #aaa; }
  .toast { position: fixed; bottom: 16px; right: 16px; background: var(--brand-deep);
           padding: 8px 12px; border-radius: 4px; display: none; }
  .toast.error { background: #7a2a2a; }
  textarea, input { background: #222; color: #eee; border: 1px solid #333;
                    border-radius: 4px; padding: 6px 8px; font: inherit; box-sizing: border-box; width: 100%; }
  textarea { min-height: 100px; font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px; }
  button { background: var(--accent); color: #fff; border: 0; border-radius: 4px;
           padding: 8px 14px; cursor: pointer; font: inherit; }
  button:hover { background: var(--accent-h); }
  button.danger { background: #7a2a2a; }
  button.danger:hover { background: #8a3434; }
  button:disabled { opacity: 0.4; cursor: not-allowed; }
  .actions { display: flex; gap: 8px; margin-top: 12px; }
  code { background: #222; padding: 1px 4px; border-radius: 2px; }
</style>
</head>
<body>
<header>
  <h1>Vecretary Memory &mdash; Score Helper</h1>
  <a href="/">&laquo; back to segments</a>
  <a href="/decay">/decay</a>
  <div style="flex:1"></div>
</header>
<main>
  <div class="panel">
    <h2>What are you saving?</h2>
    <label>Title (short)</label>
    <input id="m-title" placeholder="e.g., ComfyUI cleanup is done">
    <label style="margin-top:8px;display:block">Content (the actual memory)</label>
    <textarea id="m-content" placeholder="e.g., Confirm the target workspace before modifying files."></textarea>
  </div>

  <div class="panel">
    <h2>Q1. How long will this be relevant?</h2>
    <label class="q"><input type="radio" name="q1" value="short"> Short-term (this session / next session / a few days)</label>
    <label class="q"><input type="radio" name="q1" value="long" checked> Long-term (weeks, months, or standing)</label>
  </div>

  <div class="panel" id="p-short" style="display:none">
    <h2>Q2. How short is short-term?</h2>
    <label class="opt"><input type="radio" name="q2" value="right_now"> Right now (6 turns surface, 6 more retrievable, then gone)
      <small>Use when: "I need this in front of me for the next few turns". Saved as ephemeral.</small></label>
    <label class="opt"><input type="radio" name="q2" value="session"> This session only (fades in ~3 active hours)
      <small>Use when: relevant for the current working session, gone tomorrow. Saved as 91 + STC.</small></label>
    <label class="opt"><input type="radio" name="q2" value="next_session"> Next session too (fades in ~24 active hours)
      <small>Use when: relevant today and tomorrow. Saved as 92 + STC.</small></label>
    <label class="opt"><input type="radio" name="q2" value="multiday"> A few days (fades in ~72 active hours)
      <small>Use when: relevant for this work cycle. Saved as 93 + STC.</small></label>
  </div>

  <div class="panel" id="p-long">
    <h2>Q2. How important is this long-term?</h2>
    <label class="opt"><input type="radio" name="q2" value="reference" checked> Reference (linger a long time, slow fade)
      <small>Saved as 95. Quick drop to 88, then slow linear to floor 60 over 1000h active.</small></label>
    <label class="opt"><input type="radio" name="q2" value="important"> Important (slow decay to a low floor)
      <small>Saved as 94. Quick drop to 88, then slow linear to floor 29 over 500h active.</small></label>
    <label class="opt"><input type="radio" name="q2" value="solid"> Solid context (comes and goes with relevance)
      <small>Saved as 90. Quick drop to 89, then slow linear to floor 60 over 200h active.</small></label>
    <label class="opt"><input type="radio" name="q2" value="mandatory"> Mandatory standing instruction
      <small>Saved as 100. Session/turn reminders rise to 99 and return to 79 after model access. Use sparingly.</small></label>
  </div>

  <div class="panel">
    <div class="actions">
      <button onclick="recommend()">Recommend score</button>
      <button onclick="ephemeralQuick()">One-click ephemeral (6 turns)</button>
    </div>
  </div>

  <div class="recommend" id="recommend">
    <h3>Recommendation</h3>
    <div class="score" id="rec-score"></div>
    <div id="rec-tier"></div>
    <div class="reasoning" id="rec-reasoning" style="margin:8px 0"></div>
    <div id="rec-extra"></div>
    <div class="actions">
      <button onclick="applyRecommendation()">Save with this score</button>
    </div>
  </div>
</main>
<div class="toast" id="toast"></div>
<script>
let lastRec = null;

function toast(msg, isError) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'toast' + (isError ? ' error' : '');
  t.style.display = 'block';
  setTimeout(() => t.style.display = 'none', 2500);
}

async function api(path, opts = {}) {
  const r = await dashboardFetch(path, {
    method: opts.method || 'GET',
    headers: opts.body ? {'Content-Type': 'application/json'} : {},
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || ('HTTP ' + r.status));
  return j;
}

function getQ1() {
  const el = document.querySelector('input[name="q1"]:checked');
  return el ? el.value : 'long';
}
function getQ2() {
  const el = document.querySelector('input[name="q2"]:checked');
  return el ? el.value : 'solid';
}

function showHide() {
  const q1 = getQ1();
  document.getElementById('p-short').style.display = q1 === 'short' ? '' : 'none';
  document.getElementById('p-long').style.display  = q1 === 'long'  ? '' : 'none';
}

document.querySelectorAll('input[name="q1"]').forEach(el => {
  el.addEventListener('change', () => { showHide(); lastRec = null; document.getElementById('recommend').classList.remove('show'); });
});
document.querySelectorAll('input[name="q2"]').forEach(el => {
  el.addEventListener('change', () => { lastRec = null; document.getElementById('recommend').classList.remove('show'); });
});

async function recommend() {
  const title = document.getElementById('m-title').value.trim();
  const content = document.getElementById('m-content').value.trim();
  if (!title || !content) { toast('Title and content required', true); return; }
  try {
    const r = await api('/api/score/recommend', {
      method: 'POST',
      body: { q1: getQ1(), q2: getQ2(), title, content },
    });
    lastRec = r;
    document.getElementById('rec-score').textContent =
      'score ' + r.score + (r.decay_profile !== 'standard' ? ' + ' + r.decay_profile : '');
    document.getElementById('rec-tier').textContent = r.tier_label;
    document.getElementById('rec-reasoning').textContent = r.reasoning;
    document.getElementById('rec-extra').innerHTML = r.ephemeral
      ? '<small style="color:#fc6">⚠ Ephemeral: 6-turn surface then 6-turn fade then expired.</small>'
      : '';
    document.getElementById('recommend').classList.add('show');
  } catch (e) { toast(String(e), true); }
}

async function applyRecommendation() {
  if (!lastRec) { toast('Run recommend first', true); return; }
  const title = document.getElementById('m-title').value.trim();
  const content = document.getElementById('m-content').value.trim();
  if (!title || !content) { toast('Title and content required', true); return; }
  try {
    const r = await api('/api/add', {
      method: 'POST',
      body: {
        title, content,
        score: lastRec.score,
        segment_type: 'context',
        decay_profile: lastRec.decay_profile,
      },
    });
    toast('Saved ' + r.id.slice(0, 8));
    document.getElementById('m-title').value = '';
    document.getElementById('m-content').value = '';
  } catch (e) { toast(String(e), true); }
}

async function ephemeralQuick() {
  const title = document.getElementById('m-title').value.trim() || 'Ephemeral note';
  const content = document.getElementById('m-content').value.trim();
  if (!content) { toast('Content required', true); return; }
  try {
    const r = await api('/api/score/ephemeral', {
      method: 'POST', body: { title, content },
    });
    toast('Ephemeral saved, phase=' + r.phase + ' turns_left=' + r.turns_left);
    document.getElementById('m-title').value = '';
    document.getElementById('m-content').value = '';
  } catch (e) { toast(String(e), true); }
}

showHide();
</script>
</body>
</html>
"""






INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Vecretary Memory Dashboard</title>
<style>
  /* Charcoal surfaces with deep/bright green accents. */
  :root {
    color-scheme: dark;
    --gap: 12px;
    --radius: 8px;
    /* dark theme (default) */
    --bg:        #101512;
    --surface:   #1a211d;
    --surface-2: #232d26;
    --border:    #334239;
    --text:      #e5ede7;
    --text-dim:  #9aada0;
    --accent:    #176331;
    --accent-h:  #237943;
    --brand-bright: #5bec7d;
    --brand-deep: #246f3a;
    --success:   #1d7136;
    --success-h: #278d46;
    --danger:    #a83a2a;   /* read-only / blocked (red) */
    --danger-h:  #c44835;
  }
  body[data-theme="light"] {
    --bg:        #d9dcda;
    --surface:   #e6e8e7;
    --surface-2: #d2d7d4;
    --border:    #c9d9cd;
    --text:      #1a241d;
    --text-dim:  #4f6255;
    --accent:    #246f3a;
    --accent-h:  #287a43;
    --brand-bright: #157a34;
    --brand-deep: #285a38;
  }
  body { font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
         margin: 0; background: var(--bg); color: var(--text); }
  body { color-scheme: dark; }
  body[data-theme="light"] { color-scheme: light; }
  :focus-visible { outline: 2px solid var(--brand-bright); outline-offset: 2px; }
  /* App header: brand, primary navigation and task-focused actions. */
  header { background: var(--surface); padding: 8px 20px; border-bottom: 1px solid var(--border);
           display: grid; grid-template-columns: auto 1fr auto; align-items: center; gap: 16px;
           min-height: 56px; position: sticky; top: 0; z-index: 10;
           box-shadow: 0 1px 0 rgba(255,255,255,0.03); }
  header h1 { font-size: 15px; margin: 0; font-weight: 600; letter-spacing: -0.01em; color: var(--brand-bright); }
  header .center { display: flex; align-items: center; justify-content: center; gap: 12px; }
  header .right { display: flex; align-items: center; gap: 8px; }
  /* Main dashboard grid: overview, workspace, memory list and detail. */
  main { display: grid; grid-template-columns: 280px minmax(0,1fr) 320px;
         gap: var(--gap); padding: var(--gap); max-width: 1600px; margin: 0 auto; }
  /* Panel: more breathing room, softer corners */
  .panel { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
           padding: 16px; box-shadow: 0 1px 0 rgba(255,255,255,0.02); }
  .panel h2 { font-size: 11px; margin: 0 0 12px 0; text-transform: uppercase;
              letter-spacing: 0.08em; color: var(--brand-deep); font-weight: 600; }
  .seg-list { max-height: calc(100vh - 220px); overflow: auto; border-radius: 6px; background: var(--surface); }
  .seg-item { padding: 10px 12px; border-bottom: 1px solid var(--border); cursor: pointer;
              transition: background 0.1s ease, border-color 0.1s ease;
              border-left: 2px solid transparent; background: var(--surface); color: var(--text); }
  .seg-item:hover { background: var(--surface-2); border-left-color: var(--accent); }
  .seg-item.selected { background: var(--surface-2); border-left-color: var(--brand-bright); }
  .seg-item .t { font-weight: 600; font-size: 13px; color: var(--text); }
  .seg-item .meta { font-size: 11px; color: var(--text-dim); margin-top: 2px; }
  .seg-item .kpis { font-size: 10px; color: var(--text-dim); margin-top: 4px; display: flex; flex-wrap: wrap; gap: 4px; }
  .seg-item .kpi { background: var(--surface-2); padding: 1px 5px; border-radius: 3px; color: var(--text-dim); }
  .seg-item .kpi b { color: var(--brand-bright); }
  .seg-item .kpi.path { background: var(--surface-2); color: var(--text-dim); font-style: italic; max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .score { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 11px;
           background: var(--surface-2); color: var(--brand-bright); font-weight: 600; border: 1px solid var(--brand-deep); }
  .score.m { background: var(--danger); color: #fff; border-color: var(--danger); }
  .score.h { background: var(--brand-deep); color: #fff; border-color: var(--brand-bright); }
  .toolbar { display: flex; gap: 6px; flex-wrap: wrap; }
  /* Unified input + button baseline */
  input, textarea, select {
    background: var(--surface-2); color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 7px 10px; font: inherit;
    transition: border-color 0.1s ease, background 0.1s ease;
  }
  input, textarea, select { width: 100%; box-sizing: border-box; }
  input:focus, textarea:focus, select:focus { outline: none; border-color: var(--accent); background: var(--surface); }
  textarea { min-height: 140px; font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px; line-height: 1.5; }
  /* Unified button: same shape everywhere — only colour flips for state. */
  button {
    background: var(--accent); color: #fff; border: 1px solid var(--accent);
    border-radius: 6px; padding: 7px 12px; font: inherit; font-weight: 500;
    cursor: pointer; transition: background 0.1s ease, transform 0.08s ease, border-color 0.1s ease;
  }
  button:hover { background: var(--accent-h); border-color: var(--accent-h); transform: translateY(-1px); }
  button:active { transform: translateY(0); }
  button:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }
  button.danger { background: var(--danger); border-color: var(--danger); }
  button.danger:hover { background: var(--danger-h); border-color: var(--danger-h); }
  button.success { background: var(--success); border-color: var(--success); }
  button.success:hover { background: var(--success-h); border-color: var(--success-h); }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 8px; }
  .row.three { grid-template-columns: 1fr 1fr 1fr; }
  label { display: block; font-size: 11px; color: #999; margin-bottom: 2px; }
  .actions { display: flex; gap: 6px; margin-top: 8px; }
  .toast { position: fixed; bottom: 16px; right: 16px; background: #246f3a; padding: 8px 12px;
           border-radius: 4px; display: none; max-width: 480px;
           font-size: 13px; align-items: center; gap: 10px; }
  .toast.error { background: #7a2a2a; }
  .toast.success { background: #2a5a3a; }
  .toast .t-actions { display: inline-flex; gap: 6px; margin-left: auto; }
  .toast .t-actions button {
    background: rgba(255,255,255,.12); color: #fff; border: 1px solid rgba(255,255,255,.18);
    padding: 3px 9px; font-size: 12px; border-radius: 3px; cursor: pointer;
  }
  .toast .t-actions button:hover { background: rgba(255,255,255,.22); }
  .toast .t-actions button.danger {
    background: rgba(255,80,80,.22); border-color: rgba(255,80,80,.45);
  }
  .toast .t-actions button.danger:hover { background: rgba(255,80,80,.35); }
  .toast .t-msg { display: inline-block; vertical-align: middle; }
  /* Stacking: subsequent toasts nudge the previous one up. */
  .toast { box-shadow: 0 4px 12px rgba(0,0,0,.35); }
  .seg-detail { white-space: pre-wrap; font-family: ui-monospace, Menlo, Consolas, monospace;
                font-size: 12px; background: #161616; padding: 8px; border-radius: 4px;
                max-height: 320px; overflow: auto; }
  .stats { font-size: 12px; color: #aaa; }
  .stats b { color: #fff; }
  .empty { color: #777; font-style: italic; padding: 8px; }

  /* Main dashboard layout uses wider, predictable workspace panels. */
  .app-shell {
    max-width: 1480px;
    margin: 0 auto;
    padding: 20px clamp(14px, 2.4vw, 32px) 40px;
    display: grid;
    grid-template-columns: minmax(0, 1.35fr) minmax(320px, .85fr);
    gap: 18px;
  }
  .app-bar {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 10px clamp(14px, 2.4vw, 32px);
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    flex-wrap: wrap;
    position: sticky;
    top: 0;
    z-index: 10;
    box-shadow: 0 8px 24px rgba(0, 0, 0, .2);
    backdrop-filter: blur(12px);
  }
  .app-bar .brand {
    display: flex;
    align-items: center;
    gap: 11px;
    min-height: 50px;
    font-weight: 700;
    font-size: 16px;
    color: var(--text);
    text-decoration: none;
    white-space: nowrap;
  }
  .app-bar nav { display: flex; gap: 5px; flex-wrap: wrap; align-items: center; }
  .app-bar nav a, .app-bar nav button {
    color: var(--text-dim); text-decoration: none;
    min-height: 42px; display: inline-flex; align-items: center;
    padding: 0 13px; border-radius: 10px; font-size: 14px; font-weight: 550;
    border: 1px solid transparent; background: transparent; font-family: inherit;
    white-space: nowrap;
    transition: color .15s ease, background .15s ease, border-color .15s ease;
    cursor: pointer;
  }
  .app-bar nav a:hover, .app-bar nav button:hover { background: var(--surface-2); border-color: var(--border); color: var(--text); transform: none; }
.app-bar nav a[aria-current="page"] { color: var(--brand-bright); background: color-mix(in srgb, var(--brand-deep) 22%, var(--surface)); border-color: var(--brand-deep); }
  .header-tools {
    margin-left: auto;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 10px;
    flex-wrap: wrap;
  }
  .header-primary {
    display: inline-flex;
    min-height: 44px;
    align-items: center;
    justify-content: center;
    padding: 0 17px;
    border: 1px solid var(--accent);
    border-radius: 11px;
    background: var(--accent);
    color: #fff;
    text-decoration: none;
    font-size: 14px;
    font-weight: 650;
    box-shadow: 0 3px 10px rgba(32, 135, 61, .2);
    white-space: nowrap;
  }
  .header-primary:hover { background: var(--accent-h); border-color: var(--accent-h); color: #fff; }
  .header-tools #theme-toggle {
    min-width: 44px; min-height: 44px; padding: 0; border-radius: 10px;
    font-size: 14px;
  }
  .actions-menu { position: relative; }
  .actions-menu > summary {
    list-style: none;
    display: inline-flex;
    min-height: 44px;
    align-items: center;
    gap: 7px;
    padding: 0 13px;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--surface-2);
    color: var(--text);
    font-size: 14px;
    font-weight: 550;
    cursor: pointer;
    white-space: nowrap;
  }
  .actions-menu > summary::-webkit-details-marker { display: none; }
  .actions-menu > summary::after { content: "▾"; color: var(--brand-bright); font-size: 12px; }
  .actions-menu[open] > summary { border-color: var(--brand-deep); }
  .actions-menu-panel {
    position: absolute;
    top: calc(100% + 8px);
    right: 0;
    z-index: 30;
    display: grid;
    gap: 6px;
    width: min(260px, calc(100vw - 28px));
    padding: 9px;
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--surface);
    box-shadow: 0 14px 36px rgba(0, 0, 0, .35);
  }
  .actions-menu-panel button {
    width: 100%;
    min-height: 42px;
    padding: 0 11px;
    text-align: left;
    border-radius: 8px;
    background: transparent;
    color: var(--text);
    border: 1px solid transparent;
  }
.actions-menu-panel button:hover { background: var(--surface-2); border-color: var(--border); transform: none; }
.app-bar nav button:not(.danger):not(.success) { background: transparent; border-color: transparent; color: var(--text-dim); }
.app-bar nav button:not(.danger):not(.success):hover { background: var(--surface-2); border-color: var(--border); color: var(--text); }
.actions-menu-panel button:not(.danger):not(.success) { background: transparent; border-color: transparent; color: var(--text); }
.actions-menu-panel button:not(.danger):not(.success):hover { background: var(--surface-2); border-color: var(--border); }
  .status-bar, #segments-section {
    grid-column: 1 / -1;
  }
  .split-2 {
    grid-column: 1 / -1;
    grid-template-columns: minmax(0, .95fr) minmax(0, 1.25fr);
    gap: 18px;
  }
  #detail-section { grid-column: 1; }
  #mode-section { grid-column: 2; }
  .app-shell .section { min-height: 360px; }
  #segments-section { min-height: 480px; }
  #segments-section, #add-form-panel { scroll-margin-top: 120px; }
  #seg-list { min-height: 360px; }
  #add-form-panel textarea { min-height: 200px; }
  #detail-section #seg-detail { min-height: 360px !important; }
  #detail-section .seg-diff { min-height: 120px; }
  .app-shell > .status-bar { grid-column: 1 / -1; }
  .app-shell > #segments-section { grid-column: 1 / -1; }
  @media (max-width: 1120px) {
    .app-bar { gap: 10px; }
    .app-bar nav { order: 3; flex-basis: 100%; overflow-x: auto; flex-wrap: nowrap; padding-bottom: 2px; }
    .header-tools { margin-left: auto; }
  }
  @media (max-width: 760px) {
    .app-bar { padding: 10px 12px; }
    .app-bar nav { order: 3; flex-basis: 100%; }
    .app-bar nav a, .app-bar nav button { min-height: 40px; padding: 0 10px; font-size: 13px; }
    .header-tools { gap: 6px; }
    .header-primary { padding: 0 12px; }
    .actions-menu-panel { right: 0; }
    .app-shell { display: block; padding: 12px; }
    .app-shell > *, .app-shell .section { margin-bottom: 14px; }
    .app-shell .section { min-height: 360px; padding: 16px; }
    .app-shell .split-2 > .section { min-height: 360px; }
    #seg-list { min-height: 300px; }
    #add-form-panel textarea { min-height: 200px; }
  }
  .status-bar {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 8px 14px;
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    font-size: 12px;
    color: #aaa;
    flex-wrap: wrap;
  }
  .status-bar .path {
    font-family: ui-monospace, Menlo, Consolas, monospace;
    color: var(--brand-bright);
    background: #1a1a1a;
    padding: 2px 6px;
    border-radius: 3px;
    max-width: 480px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    direction: rtl;  /* show the END of the path, not the start, on narrow */
    text-align: left;
  }
  .status-bar .sep { color: #444; }
  .section {
    background: #161616;
    border: 1px solid #252525;
    border-radius: 8px;
    padding: 16px 18px;
  }
  .section h2 {
    display: flex; align-items: center; gap: 10px;
    font-size: 12px;
    margin: 0 0 12px 0;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: #8a96a8;
    font-weight: 600;
  }
  .section h2 .num {
    display: inline-flex;
    align-items: center; justify-content: center;
    width: 20px; height: 20px;
    background: #172d1f;
    color: var(--brand-bright);
    border-radius: 50%;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0;
  }
  .section h2 .sub {
    font-size: 11px;
    color: #666;
    text-transform: none;
    letter-spacing: 0;
    font-weight: 400;
    margin-left: auto;
  }
  .section h2 .sub a { color: var(--brand-bright); text-decoration: none; }
  .section h2 .sub a:hover { text-decoration: underline; }
  /* 2-column grid on wide viewports. Single-column on narrow. */
  .section-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
  }
  .section-row > .section { margin: 0; }
  /* Side-by-side split for score-helper + add-form on wide viewports. */
  .split-2 { display: grid; grid-template-columns: minmax(0, .95fr) minmax(0, 1.25fr); gap: 18px; }
  .split-2 > .section { margin: 0; }
  @media (max-width: 980px) {
    .split-2 { grid-template-columns: 1fr; }
  }
  /* Bulk bar + sort controls now live INSIDE the segments section,
     so they don't get separate top-level rows. */
  .seg-toolbar {
    display: flex; gap: 10px; flex-wrap: wrap;
    align-items: center;
    padding: 8px 0 12px;
    border-bottom: 1px solid #252525;
    margin-bottom: 12px;
    font-size: 12px;
  }
  .seg-toolbar label { color: #888; font-size: 11px; margin: 0; display: inline-flex; gap: 6px; align-items: center; }
  .seg-toolbar select {
    width: auto; min-width: 140px;
    padding: 4px 8px; font-size: 12px;
  }
  .seg-toolbar input[type="search"], .seg-toolbar #search {
    width: 260px; flex: 1; min-width: 180px;
    padding: 6px 10px;
  }
  .seg-toolbar .hint { color: #555; font-size: 10px; }
  /* The segment list now gets the bulk-bar toggle row above it. */
  #seg-list {
    max-height: max(240px, calc(100vh - 380px));
    overflow: auto;
    border: 1px solid #222;
    border-radius: 4px;
    background: #111;
  }
  .seg-item { padding: 10px 12px; border-bottom: 1px solid #1e1e1e; cursor: pointer; }
  .seg-item:hover { background: #1c1c1c; }
  .seg-item.selected { background: #172d1f; border-left: 3px solid #5bec7d; }
  /* Compact bulk checkbox column */
  .seg-item .bulk-cb { margin-right: 8px; transform: scale(1.05); }
  /* Narrow-viewport tweaks: hide secondary header items, collapse status. */
  @media (max-width: 760px) {
    .app-bar { padding: 8px 12px; gap: 8px; }
    .app-bar .right-cluster { gap: 6px; }
    .app-bar nav a { padding: 4px 8px; font-size: 12px; }
    .app-shell { padding: 12px 12px 24px; gap: 10px; }
    .section { padding: 12px 14px; }
    .status-bar { font-size: 11px; padding: 6px 10px; }
    .seg-toolbar input[type="search"], .seg-toolbar #search { width: 100%; }
  }

  .app-bar {
    padding: 10px clamp(14px, 2.4vw, 32px);
    gap: 18px;
    background: color-mix(in srgb, var(--surface) 94%, transparent);
    border-bottom-color: var(--border);
    box-shadow: 0 8px 24px rgba(0, 0, 0, .16);
    backdrop-filter: blur(12px);
  }
  .app-bar .brand {
    gap: 11px;
    min-height: 44px;
    color: var(--text);
    letter-spacing: .01em;
  }
  .app-bar .brand-mark {
    display: block;
    width: 50px;
    height: 50px;
    flex: 0 0 50px;
    object-fit: contain;
    padding: 3px;
    box-sizing: border-box;
    background: #050705;
    border: 1px solid rgba(116, 205, 112, .28);
    border-radius: 10px;
    filter: drop-shadow(0 2px 6px rgba(68, 189, 93, .2));
  }
  .app-bar nav a, .app-bar nav button {
    color: var(--text-dim);
    padding: 7px 12px;
    border-radius: 999px;
    transition: color .15s ease, background .15s ease, border-color .15s ease;
  }
  .app-bar nav a:hover, .app-bar nav button:hover {
    color: var(--text);
    background: var(--surface-2);
    border-color: var(--border);
  }
  .app-bar .right { gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
  .app-shell { max-width: 1480px; padding: 20px clamp(14px, 2.4vw, 32px) 40px; gap: 16px; }
  .status-bar {
    padding: 10px 14px;
    background: var(--surface);
    border-color: var(--border);
    border-radius: 10px;
    color: var(--text-dim);
    box-shadow: 0 2px 8px rgba(0, 0, 0, .08);
  }
  .status-bar .path { color: var(--brand-bright); background: var(--surface-2); }
  .status-bar .sep { color: var(--border); }
  .section {
    background: var(--surface);
    border-color: var(--border);
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 5px 18px rgba(0, 0, 0, .09);
  }
  .section h2 { color: var(--brand-deep); }
  .section h2 .num { background: var(--surface-2); color: var(--brand-bright); }
  .section h2 .sub { color: var(--text-dim); }
  .section h2 .sub a { color: var(--accent-h); }
  .seg-toolbar { border-bottom-color: var(--border); }
  .seg-toolbar label { color: var(--text-dim); }
  .seg-toolbar .hint { color: var(--text-dim); }
  #seg-list { border-color: var(--border); border-radius: 8px; background: var(--surface-2); }
  .seg-item { border-bottom-color: var(--border); background: var(--surface); }
  .seg-item:hover { background: var(--surface-2); }
  .seg-item.selected { background: color-mix(in srgb, var(--accent) 16%, var(--surface)); border-left-color: var(--brand-bright); }
  @media (max-width: 760px) {
    .app-bar { gap: 8px; }
    .app-bar .brand-mark { width: 42px; height: 42px; flex-basis: 42px; }
    .app-bar .right { justify-content: flex-start; }
    .section { padding: 14px; }
  }
  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { scroll-behavior: auto !important; transition-duration: .01ms !important; }
  }

  /* ---- Step 1 design tokens (A4 session badge + A5 health banner) ---- */
  /* These are additive — they don't override any existing rule. */
  :root {
    --ok: #5bec7d;
    --ok-bg: #172d1f;
    --warn: #d9a441;
    --warn-bg: #2a241a;
    --down: #c45858;
    --down-bg: #2a1a1a;
    --idle: #888;
    --idle-bg: #202923;
    --pill-fg: #ddd;
  }
  .pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 3px 10px; border-radius: 999px;
    font-size: 11.5px; line-height: 1.2; font-weight: 500;
    background: var(--idle-bg); color: var(--pill-fg);
    border: 1px solid #333;
    white-space: nowrap;
    transition: background 200ms, border-color 200ms;
  }
  .pill .dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--idle); flex-shrink: 0;
  }
  .pill.ok    { background: var(--ok-bg);   border-color: rgba(95,184,120,.35); }
  .pill.ok    .dot { background: var(--ok); box-shadow: 0 0 6px rgba(95,184,120,.55); }
  .pill.warn  { background: var(--warn-bg); border-color: rgba(217,164,65,.35); }
  .pill.warn  .dot { background: var(--warn); box-shadow: 0 0 6px rgba(217,164,65,.55); }
  .pill.down  { background: var(--down-bg); border-color: rgba(196,88,88,.45); }
  .pill.down  .dot { background: var(--down); box-shadow: 0 0 6px rgba(196,88,88,.55); }
  .pill.idle  { background: var(--idle-bg); border-color: #333; }
  .pill.idle  .dot { background: var(--idle); }
  /* Step 6 (D4): degraded state. Subtle but distinct from warn: a
     purple-tinged dot means the dashboard is alive but at least one
     subsystem reported a problem. Tooltip lists the reasons. */
  .pill.degraded { background: #2a1a2a; border-color: rgba(180,120,200,.45); }
  .pill.degraded .dot { background: #b478c8; box-shadow: 0 0 6px rgba(180,120,200,.55); }

  /* A4 — session badge in header. Sits flush with the right cluster. */
  #session-badge {
    cursor: default;
    user-select: none;
  }
  #session-badge .label-when {
    color: #aaa;
  }

  /* A5 — health banner between readonly-banner and main. Slim, 1-line. */
  #health-banner {
    display: none;          /* shown by JS once first poll lands */
    padding: 9px 16px;
    border-bottom: 1px solid #2a2a2a;
    font-family: ui-monospace, Menlo, Consolas, monospace;
    font-size: 12.5px;
    color: #ccc;
    background: #161616;
    display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
  }
  #health-banner.ok   { background: var(--ok-bg);   border-bottom-color: rgba(95,184,120,.30); color: #cfe9d4; }
  #health-banner.warn { background: var(--warn-bg); border-bottom-color: rgba(217,164,65,.30); color: #efdfba; }
  #health-banner.down { background: var(--down-bg); border-bottom-color: rgba(196,88,88,.35); color: #f1c8c8; }
  #health-banner .sep { color: #555; }
  #health-banner .kv b { color: #fff; font-weight: 600; }
  #health-banner .kv.muted { color: #888; }
  #health-banner #hb-refresh { margin-left: auto; }
  #health-banner #hb-refresh button {
    background: transparent; color: #888; border: 1px solid #333;
    font-size: 11px; padding: 2px 8px; border-radius: 3px;
  }
  #health-banner #hb-refresh button:hover { color: #eee; border-color: #555; }

  /* Mobile: badge shrinks; banner wraps more cleanly. */
  @media (max-width: 760px) {
    #session-badge .label-when { display: none; }
    #health-banner { font-size: 11.5px; padding: 8px 12px; }
  }

  /* ---- Step 2 — A1/A2 scorecard (Q1/Q2 + 7-box score) ---- */
  #scorecard-panel { display: flex; flex-direction: column; gap: 10px; }
  #scorecard-panel .sc-controls {
    display: grid; grid-template-columns: 1fr 1fr; gap: 10px;
  }
  #scorecard-panel .sc-q1, #scorecard-panel .sc-q2 {
    display: flex; flex-direction: column; gap: 6px;
    padding: 8px 10px; background: #161616; border: 1px solid #2a2a2a;
    border-radius: 4px;
  }
  #scorecard-panel .sc-q1 .lbl, #scorecard-panel .sc-q2 .lbl {
    font-size: 11px; color: #999; text-transform: uppercase;
    letter-spacing: .04em; margin-bottom: 2px;
  }
  #scorecard-panel .sc-pills {
    display: flex; flex-wrap: wrap; gap: 4px;
  }
  #scorecard-panel .sc-pill {
    background: #222; color: #ccc; border: 1px solid #333;
    padding: 4px 9px; border-radius: 999px; font-size: 11.5px;
    cursor: pointer; transition: all 150ms; line-height: 1.2;
  }
  #scorecard-panel .sc-pill:hover { background: #2a2a2a; border-color: #444; }
  #scorecard-panel .sc-pill.on {
    background: #172d1f; border-color: #5bec7d; color: #fff;
    box-shadow: inset 0 0 0 1px rgba(91,236,125,.25);
  }
  #scorecard-panel .sc-actions {
    display: flex; gap: 8px; align-items: center;
    padding-top: 4px; border-top: 1px dashed #2a2a2a;
  }
  #scorecard-panel .sc-actions button { font-size: 12px; }
  #scorecard-panel .sc-status {
    margin-left: auto; font-size: 11.5px; color: #888;
    font-family: ui-monospace, Menlo, Consolas, monospace;
  }

  /* The 7 scorecard boxes — grid with named cells. */
  #scorecard-panel .sc-grid {
    display: grid; grid-template-columns: repeat(7, 1fr); gap: 6px;
  }
  #scorecard-panel .sc-box {
    background: #161616; border: 1px solid #2a2a2a;
    border-radius: 5px; padding: 8px 6px; text-align: center;
    display: flex; flex-direction: column; gap: 2px; min-height: 70px;
    transition: all 200ms;
  }
  #scorecard-panel .sc-box.empty { opacity: .55; }
  #scorecard-panel .sc-box.filled { border-color: #3a4a5a; background: #1a2230; }
  #scorecard-panel .sc-box .n {
    font-size: 10px; color: #888; text-transform: uppercase;
    letter-spacing: .05em; font-weight: 600;
  }
  #scorecard-panel .sc-box .k {
    font-size: 9px; color: var(--brand-bright); text-transform: uppercase;
    letter-spacing: .04em; margin-top: 1px;
  }
  #scorecard-panel .sc-box .v {
    font-size: 15px; color: #fff; font-weight: 600; margin-top: auto;
    font-family: ui-monospace, Menlo, Consolas, monospace;
  }
  #scorecard-panel .sc-box .v small {
    font-size: 9px; color: #888; font-weight: normal; margin-left: 2px;
  }
  #scorecard-panel .sc-box .v.none { color: #777; font-weight: 400; font-size: 12px; }
  /* Color hints for the import score box (box 2). */
  #scorecard-panel .sc-box.score-high   .v { color: #c66; }
  #scorecard-panel .sc-box.score-mid    .v { color: #fc6; }
  #scorecard-panel .sc-box.score-stc    .v { color: var(--brand-bright); }
  #scorecard-panel .sc-box.score-eph    .v { color: #c9f; }
  /* Pop on/off boxes (5, 6, 7) — solid color when on, faded when off. */
  #scorecard-panel .sc-box.pop-on .v { color: var(--brand-bright); }
  #scorecard-panel .sc-box.pop-on { background: #172d1f; border-color: rgba(91,236,125,.35); }
  #scorecard-panel .sc-box.pop-off .v { color: #555; }

  #scorecard-panel .sc-reasoning {
    font-size: 11.5px; color: #aaa; line-height: 1.45;
    background: #161616; border: 1px solid #2a2a2a; border-radius: 4px;
    padding: 8px 10px; font-family: ui-monospace, Menlo, Consolas, monospace;
  }
  #scorecard-panel .sc-reasoning:empty { display: none; }
  #scorecard-panel .sc-tier {
    font-size: 12px; color: var(--brand-bright); font-weight: 600;
  }

  /* Mobile: 7 boxes become 4+3 stack to avoid cramping. */
  @media (max-width: 760px) {
    #scorecard-panel .sc-grid {
      grid-template-columns: repeat(4, 1fr);
    }
    #scorecard-panel .sc-controls {
      grid-template-columns: 1fr;
    }
  }

  /* ---- Step 3 — Recycle-bin modal (A3) ---- */
  .rb-overlay {
    position: fixed; inset: 0; background: rgba(0,0,0,.55);
    display: none; align-items: center; justify-content: center;
    z-index: 100; padding: 24px;
  }
  .rb-overlay.open { display: flex; }
  .rb-dialog {
    background: #1a1a1a; border: 1px solid #333; border-radius: 8px;
    max-width: 920px; width: 100%; max-height: 80vh;
    display: flex; flex-direction: column; overflow: hidden;
    box-shadow: 0 16px 48px rgba(0,0,0,.55);
  }
  .rb-head {
    display: flex; align-items: center; gap: 12px;
    padding: 14px 18px; border-bottom: 1px solid #2a2a2a;
    background: #1d1d1d;
  }
  .rb-head h2 {
    margin: 0; font-size: 14px; font-weight: 600;
    text-transform: uppercase; letter-spacing: .05em; color: #fc6;
  }
  .rb-head .count { color: #888; font-size: 12px; font-weight: normal; }
  .rb-head .spacer { flex: 1; }
  .rb-head button {
    background: #333; color: #ccc; border: 1px solid #444;
    padding: 5px 10px; font-size: 12px; border-radius: 3px; cursor: pointer;
  }
  .rb-head button:hover { background: #444; }
  .rb-head button.danger {
    background: #4a2020; border-color: #6a3030; color: #faa;
  }
  .rb-head button.danger:hover { background: #5a2a2a; }
  .rb-head button.primary {
    background: #246f3a; border-color: #5bec7d; color: #fff;
  }
  .rb-head button.primary:hover { background: #35b95a; }
  .rb-body { overflow: auto; padding: 8px 0; }
  .rb-empty {
    padding: 32px; text-align: center; color: #888;
    font-style: italic;
  }
  .rb-row {
    display: grid;
    grid-template-columns: 1fr 60px 80px 130px 130px 200px;
    gap: 10px; align-items: center;
    padding: 8px 18px; border-bottom: 1px solid #222;
    font-size: 12.5px;
  }
  .rb-row:hover { background: #202020; }
  .rb-row .rb-title {
    font-weight: 600; color: #eee;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  .rb-row .rb-title small {
    display: block; color: #888; font-weight: normal;
    font-size: 11px; margin-top: 1px;
  }
  .rb-row .rb-meta { color: #888; font-size: 11px; }
  .rb-row .rb-actions { display: flex; gap: 6px; justify-content: flex-end; }
  .rb-row .rb-actions button {
    background: #333; color: #ccc; border: 1px solid #444;
    padding: 3px 8px; font-size: 11.5px; border-radius: 3px; cursor: pointer;
  }
  .rb-row .rb-actions button:hover { background: #444; }
  .rb-row .rb-actions button.restore { background: #246f3a; border-color: #5bec7d; color: #e5ede7; }
  .rb-row .rb-actions button.restore:hover { background: #35b95a; }
  .rb-row .rb-actions button.purge { background: #5a2020; border-color: #7a3030; color: #fcc; }
  .rb-row .rb-actions button.purge:hover { background: #6a2828; }
  /* Mobile: stack the row. */
  @media (max-width: 760px) {
    .rb-row {
      grid-template-columns: 1fr;
      gap: 4px;
    }
    .rb-row .rb-actions { justify-content: flex-start; }
  }
  .rb-foot {
    padding: 10px 18px; border-top: 1px solid #2a2a2a;
    background: #181818; display: flex; gap: 10px; align-items: center;
    font-size: 11px; color: #888;
  }
  .rb-foot .spacer { flex: 1; }

  /* ===== Step 7 (C1) keyboard-shortcuts help overlay ===== */
  .kb-dialog { max-width: 480px; }
  .kb-body { padding: 12px 18px; }
  .kb-table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .kb-table th {
    text-align: left;
    color: #888; font-weight: 500;
    padding: 4px 6px 8px;
    border-bottom: 1px solid #2a2a2a;
    font-size: 11px; text-transform: uppercase; letter-spacing: .05em;
  }
  .kb-table td { padding: 6px; border-bottom: 1px solid #1f1f1f; }
  .kb-table td:first-child { width: 130px; white-space: nowrap; }
  .kb-table tbody tr:hover { background: #1d1d1d; }
  .kb-dialog kbd {
    display: inline-block;
    padding: 2px 6px;
    font: 11px ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    color: #ddd;
    background: #2a2a2a;
    border: 1px solid #3a3a3a;
    border-bottom-width: 2px;
    border-radius: 3px;
    vertical-align: middle;
    margin-right: 4px;
  }
  .kb-note { font-size: 11px; color: #888; margin-top: 12px; padding: 8px; background: #1a1a1a; border-radius: 4px; }
  .kb-note kbd { font-size: 10px; padding: 1px 4px; }
  /* Light-theme overrides for the help dialog */
  body[data-theme="light"] .kb-table th { color: #555; border-bottom-color: #ddd; }
  body[data-theme="light"] .kb-table td { border-bottom-color: #f0f0f0; }
  body[data-theme="light"] .kb-table tbody tr:hover { background: #f5f5f5; }
  body[data-theme="light"] .kb-dialog kbd {
    color: #222; background: #f4f4f4; border-color: #ccc;
  }
  body[data-theme="light"] .kb-note { color: #555; background: #f0f0f0; }

  /* Step 7 (C4) light theme: legacy hardcoded overrides removed —
     the body[data-theme="light"] :root variable cascade now covers
     bg/surface/border/text/accent for panels, seg-items, inputs,
     buttons, and the app bar. Only component-specific light tweaks
     (kb-table, seg-diff) remain below. */

  /* ===== Step 7 (C2) segment diff ===== */
  .seg-diff {
    margin-top: 8px;
    padding: 8px;
    background: #181818;
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    font: 12px ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    color: #ccc;
    max-height: 280px;
    overflow: auto;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .seg-diff .diff-empty { color: #666; font-style: italic; }
  .seg-diff .diff-line { display: block; padding: 1px 6px; border-radius: 2px; }
  .seg-diff .diff-added { background: #172d1f; color: #5bec7d; }
  .seg-diff .diff-removed { background: #3a1a1a; color: #c66; text-decoration: line-through; }
  .seg-diff .diff-changed { background: #3a2a1a; color: #fc6; }
  .seg-diff .diff-same { color: #888; }
  .seg-diff .diff-field { color: var(--brand-bright); font-weight: 600; }
  body[data-theme="light"] .seg-diff { background: #f8f8f8; border-color: #ddd; color: #333; }
  body[data-theme="light"] .seg-diff .diff-empty { color: #999; }
  body[data-theme="light"] .seg-diff .diff-added { background: #e8f5e8; color: #282; }
  body[data-theme="light"] .seg-diff .diff-removed { background: #f8e8e8; color: #822; }
  body[data-theme="light"] .seg-diff .diff-changed { background: #f8eed8; color: #743; }
  body[data-theme="light"] .seg-diff .diff-same { color: #aaa; }
  body[data-theme="light"] .seg-diff .diff-field { color: #247; }

  /* ===== Step 5 (B1-B7) ===== */
  /* B1: modifier-window deep link (small inline button next to score) */
  .seg-item .mod-link {
    display: inline-block;
    margin-left: 6px;
    padding: 1px 6px;
    font-size: 10px;
    font-weight: bold;
    color: var(--brand-bright);
    background: #172d1f;
    border: 1px solid #246f3a;
    border-radius: 3px;
    text-decoration: none;
    cursor: pointer;
    vertical-align: middle;
  }
  .seg-item .mod-link:hover { background: #246f3a; color: #fff; }

  /* B3: bulk-action bar (above the segment list, sticky-ish) */
  #bulk-bar {
    display: none;
    padding: 8px 12px;
    margin-bottom: 8px;
    background: #172d1f;
    border: 1px solid #246f3a;
    border-radius: 4px;
    font-family: monospace;
    font-size: 12px;
    color: #b9f5c7;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
  }
  #bulk-bar.show { display: flex; }
  #bulk-bar b { color: #5bec7d; }
  #bulk-bar button {
    background: #246f3a; color: #fff; border: 1px solid #5bec7d;
    padding: 4px 10px; font-size: 12px; cursor: pointer; border-radius: 3px;
  }
  #bulk-bar button.danger { background: #4a2a2a; color: #fcc; border-color: #8a4a4a; }
  #bulk-bar button:disabled { opacity: 0.4; cursor: not-allowed; }
  .seg-item input.bulk-cb {
    margin-right: 6px; transform: scale(1.1); cursor: pointer;
    vertical-align: middle;
  }

  /* B4: pop-audit column */
  .pop-audit {
    display: inline-block;
    margin-left: 6px;
    padding: 1px 5px;
    font-size: 10px;
    font-family: monospace;
    color: #5bec7d;
    background: #172d1f;
    border: 1px solid #246f3a;
    border-radius: 3px;
    vertical-align: middle;
  }
  .pop-audit.zero { color: #666; background: transparent; border-color: #333; }
  .pop-audit a { color: inherit; text-decoration: none; }
  .pop-audit a:hover { color: #fff; }

  /* B5: inline decay sparkline */
  .sparkline {
    display: inline-block;
    width: 60px;
    height: 16px;
    vertical-align: middle;
    margin-left: 6px;
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 2px;
  }

  /* B7: sort controls */
  #sort-controls {
    display: flex;
    gap: 8px;
    align-items: center;
    padding: 6px 12px;
    margin-bottom: 6px;
    font-family: monospace;
    font-size: 12px;
    color: #888;
    flex-wrap: wrap;
  }
  #sort-controls select {
    background: #232d26; color: var(--brand-bright); border: 1px solid #334239;
    padding: 2px 6px; font-size: 12px; border-radius: 3px;
  }
  #sort-controls label { color: #888; font-size: 11px; }
  #sort-controls .hint { color: #666; font-size: 11px; }

  /* B6: score-helper CTA chip (in add form) */
  .score-cta {
    display: inline-block;
    margin-left: 6px;
    padding: 1px 7px;
    font-size: 10px;
    font-weight: bold;
    color: #5bec7d;
    background: #172d1f;
    border: 1px solid #246f3a;
    border-radius: 3px;
    text-decoration: none;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    vertical-align: middle;
  }
  .score-cta:hover { background: #246f3a; color: #fff; }
  .status-bar .status-refresh {
    min-height: 40px;
    padding: 0 13px;
    font-size: 13px;
    white-space: nowrap;
  }
  @media (max-width: 760px) {
    .app-bar .header-tools {
      order: 2;
      flex: 1 0 100%;
      width: 100%;
      margin-left: 0;
      justify-content: flex-start;
    }
    .app-bar nav {
      order: 3;
      flex: 1 0 100%;
      min-width: 0;
      overflow-x: auto;
      flex-wrap: nowrap;
    }
    .app-bar nav a, .app-bar nav button {
      min-height: 44px;
      padding: 0 12px;
      font-size: 14px;
    }
    .header-primary, .actions-menu > summary {
      min-height: 44px;
    }
    body header.app-bar #recycle-link,
    body header.app-bar #theme-toggle {
      width: 44px;
      min-width: 44px;
      height: 44px;
      min-height: 44px;
      padding: 0;
    }
    .app-shell .section { min-height: 360px; }
    #segments-section { min-height: 480px; }
    #seg-list { min-height: 360px; }
    #detail-section #seg-detail { min-height: 360px !important; }
  }
</style>
</head>
<body>
<header class="app-bar" role="banner">
  <a href="/" class="brand" title="Vecretary Memory Dashboard — home">
    <img class="brand-mark" src="/assets/vecretary-logo-256.png" alt="Vecretary Memory">
  </a>
  <nav class="primary-nav" aria-label="Primary dashboard">
    <a href="/" aria-current="page">Overview</a>
    <a href="#segments-section">Memories</a>
    <a href="/score">Score helper</a>
    <a href="/decay">Decay</a>
    <button type="button" class="nav-action" id="recycle-link" aria-label="Recycle bin" title="Recycle bin" aria-haspopup="dialog" aria-controls="rb-overlay" onclick="loadRecycled()">♻</button>
  </nav>
  <div class="header-tools" role="group" aria-label="Dashboard actions">
    <span id="session-badge" class="pill idle" title="Session clock: drives decay tick + pop math. Updated on every tool call and dashboard interaction.">
      <span class="dot"></span><span class="label">session</span>
    </span>
    <a class="header-primary" href="#add-form-panel">+ New memory</a>
    <button id="theme-toggle" onclick="toggleTheme()" aria-label="Toggle color theme" title="Toggle light / dark theme (keyboard: t)">☀</button>
    <details class="actions-menu">
      <summary>More actions</summary>
      <div class="actions-menu-panel" role="group" aria-label="Additional dashboard actions">
        <button type="button" id="export-btn" onclick="exportJson()" title="Export all segments as JSON (keyboard: x)">Export JSON</button>
        <button type="button" id="pop-search-btn" onclick="onPopSearch()" title="Audit pop activity on the current search result">Pop audit</button>
        <button type="button" onclick="pullMandatory()" title="Simulate retrieving every mandatory (score 99-100) segment">Pull mandatory</button>
        <button type="button" onclick="runNudge()" title="Check due mandatory elevations without advancing the turn counter">Nudge check</button>
        <button type="button" onclick="loadList()" title="Re-fetch the segment list from the store">Refresh list</button>
      </div>
    </details>
  </div>
</header>

<div id="readonly-banner" style="display:none;padding:10px 16px;background:#3a2a2a;border-bottom:2px solid #b03030;color:#fcc;font-family:monospace;font-size:13px">
  <b>⚠ READ-ONLY mode</b> — this SQLite service was started as an observer.
  Browse, search, and view KPIs are enabled. <b>Writes are disabled</b> (Save / Delete / Update / Add / Pop / Tick will return 409).
  Start a separate owner SQLite service only after reviewing the current writer/session state.
</div>

<div id="search-banner" style="display:none;padding:10px 16px;background:#172d1f;border-bottom:1px solid #246f3a;color:#b9f5c7;font-family:monospace;font-size:13px"></div>

<main class="app-shell">

  <!-- Status bar: live health + stats + persist-dir -->
  <div class="status-bar" aria-label="Dashboard status">
    <span id="health-banner" aria-live="polite" style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
      <span id="hb-dot" class="pill idle"><span class="dot"></span>checking…</span>
      <span class="sep">·</span>
      <span class="kv"><b id="hb-segs">–</b> segments</span>
      <span class="kv"><b id="hb-mand">–</b> mandatory</span>
      <span class="sep">·</span>
      <span class="kv muted" id="hb-lock">lock ?</span>
      <span class="kv muted" id="hb-embed">embed ?</span>
      <span class="kv muted" id="hb-pid">pid ?</span>
    </span>
    <button class="status-refresh" type="button" onclick="pollHealthNow()" title="Refresh health and session status now">Refresh status</button>
    <span class="path" id="stats" title="On-disk store path">loading…</span>
  </div>

  <!-- SECTION 1+2: Score helper + Add segment (side-by-side on wide) -->
  <div class="split-2">
    <section class="section" id="scorecard-panel" aria-label="Score helper">
      <h2><span class="num">1</span> Score helper
        <span class="sub"><a href="/score" title="Open the full standalone score helper page">open full page →</a></span>
      </h2>
      <div class="sc-controls">
        <div class="sc-q1">
          <span class="lbl">Q1 — relevance horizon</span>
          <div class="sc-pills" id="sc-q1-pills">
            <span class="sc-pill" data-q1="short">Short-term</span>
            <span class="sc-pill on" data-q1="long">Long-term</span>
          </div>
        </div>
        <div class="sc-q2" id="sc-q2-short" style="display:none">
          <span class="lbl">Q2 — how short?</span>
          <div class="sc-pills" id="sc-q2-short-pills">
            <span class="sc-pill" data-q2="right_now">Right now (ephemeral)</span>
            <span class="sc-pill on" data-q2="session">This session</span>
            <span class="sc-pill" data-q2="next_session">Next session</span>
            <span class="sc-pill" data-q2="multiday">A few days</span>
          </div>
        </div>
        <div class="sc-q2" id="sc-q2-long">
          <span class="lbl">Q2 — how important?</span>
          <div class="sc-pills" id="sc-q2-long-pills">
            <span class="sc-pill" data-q2="reference">Reference</span>
            <span class="sc-pill" data-q2="important">Important</span>
            <span class="sc-pill on" data-q2="solid">Solid</span>
            <span class="sc-pill" data-q2="mandatory">Mandatory</span>
          </div>
        </div>
      </div>
      <div class="sc-actions">
        <button type="button" id="sc-recommend-btn" title="Compute a score from your Q1 (semantic match) and Q2 (durability) answers">⚡ Recommend score</button>
        <button type="button" id="sc-apply-btn" disabled title="Copy the recommended score into the form's score field">↓ Apply to form</button>
        <button type="button" id="sc-reset-btn" title="Clear the scorecard back to the initial empty state">Reset</button>
        <span class="status" id="sc-status">pick answers above</span>
      </div>
      <div class="sc-grid" id="sc-grid">
        <div class="sc-box empty pop-off" data-box="1"><div class="v none">—</div><div class="lbl">SEM</div></div>
        <div class="sc-box empty pop-off" data-box="2"><div class="v none">—</div><div class="lbl">IMP</div></div>
        <div class="sc-box empty pop-off" data-box="3"><div class="v none">—</div><div class="lbl">DROP</div></div>
        <div class="sc-box empty pop-off" data-box="4"><div class="v none">—</div><div class="lbl">MOD</div></div>
        <div class="sc-box empty pop-off" data-box="5"><div class="v none">—</div><div class="lbl">POP1</div></div>
        <div class="sc-box empty pop-off" data-box="6"><div class="v none">—</div><div class="lbl">POP2</div></div>
        <div class="sc-box empty pop-off" data-box="7"><div class="v none">—</div><div class="lbl">KW</div></div>
      </div>
      <div class="sc-reasoning" id="sc-reasoning"></div>
    </section>
    <section class="section" id="add-form-panel" aria-label="Add or edit segment">
      <h2><span class="num">2</span> <span id="form-title">Add segment</span>
        <span class="sub"><a href="/score" title="Open the score helper to choose a score">/score helper</a></span>
      </h2>
      <form id="seg-form" onsubmit="return onSubmit(event)">
        <input type="hidden" id="f-id">
        <div class="row">
          <div><label>Title</label><input id="f-title" required title="Short label for the concept this memory represents (e.g. &quot;Pinecone auth flow&quot;)"></div>
          <div><label>Project</label><input id="f-project" placeholder="optional" title="Optional grouping tag (e.g. project name). Filters in /api/list."></div>
        </div>
        <div class="row three">
          <div><label>Score (0-100) <a class="score-cta" href="/score" title="Open the score helper">helper</a></label><input id="f-score" type="number" min="0" max="100" value="50" title="0-100 importance. 99-100 = mandatory lifecycle. 90/94/95 = special standard tiers. 80-89 = plateau; other standard scores 91-98 use its fallback. 50-79 = mid. 0-49 = tail. Profiles override standard decay. Use /score helper above for guidance."></div>
          <div><label>Type</label>
            <select id="f-type" title="Category — affects decay modifier + pop priority. standing_instruction and critical_constraint auto-promote to 100.">
              <option value="context">context</option>
              <option value="standing_instruction">standing instruction (mandatory)</option>
              <option value="critical_constraint">critical constraint (mandatory)</option>
              <option value="architectural_decision">architectural decision</option>
              <option value="implementation_detail">implementation detail</option>
              <option value="exploration">exploration</option>
              <option value="question">question</option>
              <option value="answer">answer</option>
              <option value="code">code</option>
              <option value="error_resolution">error resolution</option>
              <option value="user_preference">user preference</option>
            </select>
          </div>
          <div><label>Tags (comma-sep)</label><input id="f-tags" placeholder="optional" title="Free-form tags. Comma-separated. Useful for filtering in /api/list."></div>
        </div>
        <div class="row three">
          <div><label>Decay profile</label>
            <select id="f-profile" title="How aggressively this memory fades. standard = tiers 90/94/95 + plateau/mid/tail. short_term_critical = STC 91/92/93 with 3h/24h/72h active fade. ephemeral = score 80, surfaces then expires over 6+6 turns.">
              <option value="standard">standard (default — tiers 90/94/95, plateau, mid, tail)</option>
              <option value="short_term_critical">short-term critical (STC 91/92/93 — 3h/24h/72h active fade)</option>
              <option value="ephemeral">ephemeral (score 80 — surface → fade → expire per turn counter)</option>
            </select>
          </div>
          <div><label>Bump count</label><input id="f-bump-count" disabled placeholder="0" title="Number of times the pop engine has surfaced this memory (lifetime). 0 if never popped."></div>
          <div><label>Last bumped</label><input id="f-last-bump" disabled placeholder="never" title="When the pop engine last bumped this memory. Empty if never."></div>
        </div>
        <label>Content</label>
        <textarea id="f-content" required title="The body of the memory. This is what the retriever matches against — be specific and self-contained."></textarea>
        <div class="actions">
          <button type="submit" id="save-btn" title="Persist the segment to the store (or update if editing an existing one)">Save</button>
          <button type="button" id="new-btn" onclick="resetForm()" title="Clear the form and start a new segment (discards unsaved edits)">New</button>
          <button type="button" class="danger" id="del-btn" style="display:none" onclick="onRecycle()" title="Move this segment to the recycle bin (soft delete; recoverable)">♻ Recycle</button>
          <button type="button" class="danger" id="purge-btn" style="display:none" onclick="onPurge()" title="HARD delete — irreversible. Only use from the recycle bin view.">✖ Purge (hard)</button>
          <button type="button" id="restore-btn" style="display:none;background:#3a5a3a;color:#cfc" onclick="onRestore()" title="Restore this segment from the recycle bin">↺ Restore</button>
        </div>
      </form>
    </section>
  </div>

  <!-- SECTION 3: Segments (toolbar + bulk bar + list + detail) -->
  <section class="section" id="segments-section" aria-label="Segments">
    <h2><span class="num">3</span> Segments <span id="listcount"></span>
      <span class="sub"><span id="bulk-count-pill" style="display:none"><b id="bulk-count">0</b> selected</span></span>
    </h2>
    <div class="seg-toolbar" role="toolbar" aria-label="Segment list toolbar">
      <select id="search-mode" title="Choose which search to run when you type in the search box">
        <option value="filter">Local filter</option>
        <option value="retriever" selected>⚡ Retriever search</option>
      </select>
      <input id="search" type="search" placeholder="search…" title="Type to filter the list (local mode) or run a retriever search (agent mode). Enter = force-run. Esc = clear.">
      <label for="sort-key">Sort:</label>
      <select id="sort-key" title="Primary sort key for the segment list">
        <option value="score-desc" selected>Score ↓</option>
        <option value="score-asc">Score ↑</option>
        <option value="accessed-desc">Accessed ↓</option>
        <option value="accessed-asc">Accessed ↑</option>
        <option value="popped-desc">Pop count ↓</option>
        <option value="title-asc">Title A→Z</option>
      </select>
      <label><input type="checkbox" id="sort-mandatory-first" checked> Mandatory first</label>
      <span class="hint">filters + sort persist in localStorage</span>
    </div>
    <div id="bulk-bar" role="toolbar" aria-label="Bulk actions for selected segments">
      <b id="bulk-count-row">0</b> selected
      <button onclick="bulkBumpScore(10)" title="+10 to every selected segment">+10 score</button>
      <button onclick="bulkBumpScore(-10)" title="-10 from every selected segment">−10 score</button>
      <button onclick="bulkSetScore()" title="Set absolute score on every selected segment">set score…</button>
      <button class="danger" onclick="bulkRecycle()" title="Move all selected to the recycle bin">♻ recycle selected</button>
      <span style="flex:1"></span>
      <button onclick="bulkClear()" title="Deselect every segment and hide the bulk action bar">Clear selection</button>
    </div>
    <div id="seg-list" class="seg-list" aria-label="Segment list"></div>
  </section>

  <!-- SECTION 4: Detail (selected segment details) -->
  <section class="section" id="detail-section" aria-label="Selected segment detail">
    <h2><span class="num">4</span> Detail
      <span class="sub">stored JSON</span>
    </h2>
    <div id="nudge-controls" style="display:none">
      <label>Nudge interval (turns; 0 disables)<input id="nudge-interval" type="number" min="0" step="1"></label>
      <label><input id="nudge-active" type="checkbox"> Nudge active</label>
      <button onclick="saveNudge()">Save nudge settings</button>
    </div>
    <div id="seg-detail" class="seg-detail" style="min-height:120px">(no segment selected)</div>
    <!-- Step 7 (C2) — segment diff view (stored vs current edit-form state) -->
    <div id="seg-diff" class="seg-diff" aria-live="polite" aria-label="Pending edits compared to stored values">
      <div class="diff-empty">Edit the form above to see live diffs vs. the stored segment.</div>
    </div>
  </section>

  <!-- SECTION 5: Single SQLite service ownership status -->
  <section class="section" id="mode-section" aria-label="Dashboard mode">
    <h2><span class="num">5</span> Mode
      <span class="sub" id="mode-status">checking mode…</span>
    </h2>
    <p>MCP and dashboard requests share the same SQLite service owner. Separate writer handoff is not available.</p>
  </section>
</main>

<!-- Step 3 A3 — Recycle-bin modal. Hidden by default; loadRecycled() opens it. -->
<div class="rb-overlay" id="rb-overlay" role="dialog" aria-modal="true" aria-labelledby="rb-title">
  <div class="rb-dialog">
    <div class="rb-head">
      <h2 id="rb-title">♻ Recycle bin</h2>
      <span class="count" id="rb-count"></span>
      <span class="spacer"></span>
      <button class="primary" id="rb-restore-all" title="Restore every recycled segment back to the active set">↺ Restore all</button>
      <button class="danger" id="rb-empty-bin" title="HARD delete every segment in the recycle bin. IRREVERSIBLE.">✖ Empty bin</button>
      <button id="rb-close" title="Close (Esc)">×</button>
    </div>
    <div class="rb-body" id="rb-body"></div>
    <div class="rb-foot">
      <span>Soft-deleted segments. Restore brings them back; Purge hard-deletes.</span>
      <span class="spacer"></span>
      <span>Recycled items are excluded from search and the retriever.</span>
    </div>
  </div>
</div>

<!-- Step 7 (C1) — Keyboard shortcuts help overlay. Hidden by default; press ? to open. -->
<div class="rb-overlay" id="kb-overlay" role="dialog" aria-modal="true" aria-labelledby="kb-title" style="display:none">
  <div class="rb-dialog kb-dialog">
    <div class="rb-head">
      <h2 id="kb-title">⌨ Keyboard shortcuts</h2>
      <span class="spacer"></span>
      <button id="kb-close" title="Close (Esc)">×</button>
    </div>
    <div class="rb-body kb-body">
      <table class="kb-table">
        <thead>
          <tr><th>Key</th><th>Action</th></tr>
        </thead>
        <tbody>
          <tr><td><kbd>?</kbd></td><td>Open this help</td></tr>
          <tr><td><kbd>Esc</kbd></td><td>Close help / modal / unfocus search</td></tr>
          <tr><td><kbd>/</kbd></td><td>Focus the search box</td></tr>
          <tr><td><kbd>j</kbd> / <kbd>↓</kbd></td><td>Next segment in the list</td></tr>
          <tr><td><kbd>k</kbd> / <kbd>↑</kbd></td><td>Previous segment in the list</td></tr>
          <tr><td><kbd>n</kbd></td><td>Focus the new-segment form (clears current selection)</td></tr>
          <tr><td><kbd>e</kbd></td><td>Focus the edit form (when a segment is selected)</td></tr>
          <tr><td><kbd>r</kbd></td><td>Recycle the selected segment (asks to confirm)</td></tr>
          <tr><td><kbd>t</kbd></td><td>Toggle light / dark theme</td></tr>
          <tr><td><kbd>x</kbd></td><td>Open JSON export of all segments</td></tr>
        </tbody>
      </table>
      <p class="kb-note">Shortcuts are disabled when typing in a text input or
      textarea (except <kbd>Esc</kbd>, which always works).</p>
    </div>
  </div>
</div>

<div class="toast" id="toast" style="display:none" aria-hidden="true"></div>
<script>
let allSegs = [];

// Toast — small status notification bottom-right.
// Signature: toast(msg, opts?)
//   opts.error     → red error pill (default false)
//   opts.success   → green success pill (default false)
//   opts.duration  → ms before auto-dismiss; 0 = sticky (default 2500)
//   opts.actions   → array of {label, onClick, danger?} buttons
//   opts.id        → dedup; if same id already showing, replaced in place
// Backward-compatible: toast('x', true) still works as before.
let _toastCounter = 0;
function toast(msg, opts) {
  if (typeof opts === 'boolean') opts = { error: opts };
  opts = opts || {};
  const id = opts.id || ('__toast_' + (++_toastCounter));

  // Reuse the existing #toast node for the simple case (back-compat
  // callers expect toast.textContent = '...'), but create a fresh
  // node when actions are present so layout is correct.
  let t = document.getElementById(id);
  if (!t) {
    t = document.createElement('div');
    t.id = id;
    t.className = 'toast';
    document.body.appendChild(t);
  }
  // Cancel any pending dismiss timer from the previous use of this node.
  if (t._dismissTimer) { clearTimeout(t._dismissTimer); t._dismissTimer = null; }

  t.className = 'toast' + (opts.error ? ' error' : opts.success ? ' success' : '');
  t.style.display = 'block';
  t.setAttribute('role', 'status');
  t.setAttribute('aria-live', 'polite');
  t.setAttribute('aria-hidden', 'false');
  t.innerHTML = '';
  const msgSpan = document.createElement('span');
  msgSpan.className = 't-msg';
  msgSpan.textContent = msg;
  t.appendChild(msgSpan);

  if (opts.actions && opts.actions.length) {
    const acts = document.createElement('span');
    acts.className = 't-actions';
    for (const a of opts.actions) {
      const b = document.createElement('button');
      b.textContent = a.label;
      if (a.danger) b.classList.add('danger');
      b.addEventListener('click', async (ev) => {
        ev.stopPropagation();
        try { await a.onClick(); } finally {
          dismiss();
        }
      });
      acts.appendChild(b);
    }
    t.appendChild(acts);
  }

  // Always start the timer fresh, even if we're reusing the node.
  const dur = opts.duration != null ? opts.duration : 2500;
  if (dur > 0) {
    t._dismissTimer = setTimeout(dismiss, dur);
  }

  function dismiss() {
    if (t && t.parentNode) t.parentNode.removeChild(t);
    t._dismissTimer = null;
  }
  return t;
}

async function api(path, opts = {}) {
  const r = await dashboardFetch(path, {
    method: opts.method || 'GET',
    headers: opts.body ? {'Content-Type': 'application/json'} : {},
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || ('HTTP ' + r.status));
  return j;
}

function scoreClass(s) { return s >= 95 ? ' m' : (s >= 80 ? ' h' : ''); }

function renderKpiRow(s) {
  const k = s.kpis || {};
  if (!k || k.error) return '';
  const init = k.initial_score != null ? k.initial_score : '-';
  const cur = k.current_score != null ? k.current_score : s.score;
  const overall = k.overall_score != null ? k.overall_score.toFixed(1) : '-';
  const actH = k.active_hours_elapsed != null ? k.active_hours_elapsed.toFixed(1) : '-';
  const effH = k.effective_hours_since_pop != null ? k.effective_hours_since_pop.toFixed(1) : '-';
  const kw = k.keyword_hits_total != null ? k.keyword_hits_total : 0;
  const acc = k.accesses_last_24h_active != null ? k.accesses_last_24h_active : 0;
  const pops = k.pop_recent_8h_active != null ? k.pop_recent_8h_active : 0;
  const promo = k.promotions_applied != null ? k.promotions_applied : 0;
  const path = (k.decay_path_description || '').slice(0, 60);
  return `
    <div class="kpis">
      <span class="kpi" title="Initial (immutable) importance score"><b>init</b> ${init}</span>
      <span class="kpi" title="Current importance score after decay"><b>cur</b> ${cur}</span>
      <span class="kpi" title="Overall = (initial + current) / 2"><b>overall</b> ${overall}</span>
      <span class="kpi" title="Active hours elapsed since last action"><b>act</b> ${actH}h</span>
      <span class="kpi" title="Effective hours since last pop/reset"><b>eff</b> ${effH}h</span>
      <span class="kpi" title="Total keyword hits, all time"><b>kw</b> ${kw}</span>
      <span class="kpi" title="Accesses in last 24 active hours"><b>24h</b> ${acc}</span>
      <span class="kpi" title="Pops in last 8 active hours"><b>pops8h</b> ${pops}</span>
      <span class="kpi" title="Times the +10 promotion rule fired"><b>promo</b> ${promo}</span>
      <span class="kpi path" title="${escapeHtml(k.decay_path_description || '')}">${escapeHtml(path)}</span>
    </div>
  `;
}

function renderList(segs) {
  const root = document.getElementById('seg-list');
  document.getElementById('listcount').textContent = '(' + segs.length + ')';
  if (!segs.length) { root.innerHTML = '<div class="empty">no segments</div>'; return; }
  root.innerHTML = segs.map(s => {
    // B4: pop-audit badge — shows total pop count and links to /modifier
    // (where the full pop log + modifier breakdown lives). Prefers
    // the lifetime bump_count (top-level on the segment); falls back
    // to kpis for older responses.
    const popCount = (s.bump_count != null)
      ? s.bump_count
      : ((s.kpis && s.kpis.pop_count_total != null) ? s.kpis.pop_count_total : 0);
    const popBadge = popCount > 0
      ? `<a class="pop-audit" href="/modifier?id=${encodeURIComponent(s.id)}" title="Pop history + modifier breakdown" onclick="event.stopPropagation()">pop×${popCount}</a>`
      : `<span class="pop-audit zero" title="No pops on record">pop×0</span>`;
    // B5: inline sparkline (60×16 px) — server-derived decay preview.
    const spark = renderSparkline(s);
    return `
    <div class="seg-item" data-id="${escapeHtml(s.id)}" onclick="selectSeg(this.dataset.id)">
      <div class="t">
        <input type="checkbox" class="bulk-cb" data-id="${escapeHtml(s.id)}" onclick="event.stopPropagation(); onBulkToggle()" title="Select for bulk actions">
        ${escapeHtml(s.title)}
      </div>
      <div class="meta">
        <span class="score${scoreClass(s.score)}">${s.score}</span>
        <!-- B1: deep link to /modifier?id=… for the per-segment modifier breakdown -->
        <a class="mod-link" href="/modifier?id=${encodeURIComponent(s.id)}" title="Open modifier window for this segment" onclick="event.stopPropagation()">mod ↗</a>
        ${popBadge}
        ${spark}
        ${escapeHtml(s.segment_type)}
        ${s.project ? ' · ' + escapeHtml(s.project) : ''}
        ${s.tags && s.tags.length ? ' · ' + s.tags.map(escapeHtml).join(',') : ''}
      </div>
      ${renderKpiRow(s)}
    </div>
  `;
  }).join('');
  // After re-render, re-apply selection state + bulk bar visibility.
  applyBulkState();
}

// B5: inline decay sparkline. We have no per-segment time-series in
// /api/list, so we approximate from KPIs (current vs initial) with a
// small inline SVG that shows the rough decay slope. Cheap, no extra
// fetch. Click-through goes to /decay?id=… for the full chart.
function renderSparkline(s) {
  const k = s.kpis || {};
  const init = (k.initial_score != null) ? k.initial_score : (s.score || 0);
  const cur = (k.current_score != null) ? k.current_score : (s.score || 0);
  const eff = (k.effective_hours_since_pop != null) ? k.effective_hours_since_pop : 0;
  // Simple two-point polyline from initial (top-left) to current (right).
  // If current > initial (a pop happened), draw a +bump at the right.
  const w = 60, h = 16;
  const x1 = 2, x2 = w - 2;
  const yMax = 100, yMin = 0;
  const y1 = h - ((init - yMin) / (yMax - yMin)) * (h - 2) - 1;
  const y2 = h - ((cur - yMin) / (yMax - yMin)) * (h - 2) - 1;
  const stroke = (cur < init) ? 'var(--brand-bright)' : (cur > init ? '#fa6' : '#888');
  const title = `Decay preview: ${init} → ${cur} over ${eff.toFixed(1)}h active (click for full chart)`;
  return `<a href="/decay?id=${encodeURIComponent(s.id)}" title="${escapeHtml(title)}" onclick="event.stopPropagation()"><svg class="sparkline" viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg"><polyline points="${x1},${y1} ${x2},${y2}" fill="none" stroke="${stroke}" stroke-width="1.2"/></svg></a>`;
}

function escapeHtml(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, c => (
    {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]
  ));
}

// ---- Step 2 — Score helper (A1/A2 scorecard) ----
// State: which q1 + q2 pill is currently selected. Mirrors the
// standalone /score helper page but lives inline in the add/edit form.
const SC = {
  q1: 'long',         // 'short' | 'long'
  q2: 'solid',        // branch-specific
  lastRec: null,      // last recommendation payload from /api/score/recommend
};

function scSelectPills(group, value) {
  document.querySelectorAll(`#sc-${group}-pills .sc-pill, #sc-${group}-short-pills .sc-pill, #sc-${group}-long-pills .sc-pill`)
    .forEach(p => p.classList.toggle('on', p.dataset[group] === value));
}

function scGetSelected(group) {
  // group is 'q1' or 'q2'; find the pill with the matching data-q* attr.
  const sel = document.querySelector(`#scorecard-panel .sc-pill.on[data-${group}]`);
  return sel ? sel.dataset[group] : null;
}

function scShowHide() {
  const q1 = SC.q1;
  document.getElementById('sc-q2-short').style.display = q1 === 'short' ? 'flex' : 'none';
  document.getElementById('sc-q2-long').style.display  = q1 === 'long'  ? 'flex' : 'none';
}

function scReset() {
  SC.lastRec = null;
  document.getElementById('sc-reasoning').textContent = '';
  document.getElementById('sc-status').textContent = 'pick answers above';
  document.getElementById('sc-apply-btn').disabled = true;
  for (const box of document.querySelectorAll('.sc-box')) {
    box.classList.remove('filled', 'pop-on');
    box.classList.add('empty', 'pop-off');
    box.classList.remove('score-high', 'score-mid', 'score-stc', 'score-eph');
    box.querySelector('.v').className = 'v none';
    box.querySelector('.v').textContent = '—';
  }
}

function scRender(rec) {
  const sc = rec.scorecard || {};
  // Box 1: semantic
  scSetBox(1, sc.semantic || '—', false, null);
  // Box 2: importance (with color hint)
  const score = rec.score ?? sc.importance ?? 0;
  let scoreCls = 'score-mid';
  if (rec.decay_profile === 'ephemeral') scoreCls = 'score-eph';
  else if (rec.decay_profile === 'short_term_critical') scoreCls = 'score-stc';
  else if (score >= 94) scoreCls = 'score-high';
  scSetBox(2, String(score), true, scoreCls);
  // Box 3: time to quick drop
  const drop = sc.time_to_quick_drop_hours;
  scSetBox(3, drop == null ? 'none' : `${drop}h`, drop == null, null);
  // Box 4: decay modifier (short label)
  const modLabel = {
    standard: 'standard',
    short_term_critical: 'STC',
    ephemeral: 'ephemeral',
  }[sc.decay_modifier || rec.decay_profile] || '—';
  scSetBox(4, modLabel, false, null);
  // Box 5: kwpop lvl1
  scSetPop(5, sc.kwpop_lvl1 ?? 0);
  // Box 6: kwpop lvl2
  scSetPop(6, sc.kwpop_lvl2 ?? 0);
  // Box 7: kw mod initial
  const kw = sc.kw_mod_initial ?? 0;
  const kwText = kw > 0 ? `+${kw}` : '+0';
  scSetBox(7, kwText, kw > 0, 'pop-on');

  document.getElementById('sc-reasoning').innerHTML =
    `<div class="sc-tier">${escapeHtml(rec.tier_label || '')}</div>${escapeHtml(rec.reasoning || '')}`;
  document.getElementById('sc-status').textContent =
    rec.ephemeral ? 'ephemeral · 6+6 turns' : 'recommended';
  document.getElementById('sc-apply-btn').disabled = false;
  SC.lastRec = rec;
}

function scSetBox(n, value, filled, extraCls) {
  const box = document.querySelector(`.sc-box[data-box="${n}"]`);
  if (!box) return;
  box.classList.remove('empty', 'filled', 'score-high', 'score-mid', 'score-stc', 'score-eph');
  box.classList.add(filled ? 'filled' : 'empty');
  if (extraCls) box.classList.add(extraCls);
  const v = box.querySelector('.v');
  v.className = 'v' + (filled ? '' : ' none');
  v.textContent = value;
}

function scSetPop(n, on) {
  const box = document.querySelector(`.sc-box[data-box="${n}"]`);
  if (!box) return;
  box.classList.remove('empty', 'filled', 'pop-on', 'pop-off');
  box.classList.add('filled');
  box.classList.add(on ? 'pop-on' : 'pop-off');
  const v = box.querySelector('.v');
  v.className = 'v';
  v.textContent = on ? 'on' : 'off';
}

async function scRecommend() {
  const status = document.getElementById('sc-status');
  // Read title/content from the form (same source /score helper uses).
  const title = document.getElementById('f-title').value.trim();
  const content = document.getElementById('f-content').value.trim();
  if (!title || !content) {
    status.textContent = '⚠ fill Title + Content below first';
    status.style.color = '#fc6';
    setTimeout(() => { status.style.color = ''; }, 2500);
    return;
  }
  status.textContent = 'computing…';
  try {
    const rec = await api('/api/score/recommend', {
      method: 'POST',
      body: { q1: SC.q1, q2: SC.q2, title, content },
    });
    scRender(rec);
  } catch (e) {
    status.textContent = '⚠ ' + String(e).slice(0, 60);
    status.style.color = '#c66';
    setTimeout(() => { status.style.color = ''; }, 3000);
  }
}

function scApply() {
  if (!SC.lastRec) return;
  const rec = SC.lastRec;
  // Fill the form fields the scorecard covers.
  document.getElementById('f-score').value = rec.score;
  // Map decay_profile to the form's decay-profile select value.
  const profileMap = {
    'standard': 'standard',
    'short_term_critical': 'short_term_critical',
    'ephemeral': 'ephemeral',
  };
  const sel = document.getElementById('f-profile');
  const want = profileMap[rec.decay_profile] || 'standard';
  if ([...sel.options].some(o => o.value === want)) sel.value = want;
  // Visual confirmation.
  document.getElementById('sc-status').textContent = `✓ applied (score=${rec.score}, profile=${want})`;
  // Cross-page toast so the user notices even if they're scrolled away.
  toast(`⚡ Applied to form: score=${rec.score}, profile=${want}`, { success: true });
  setTimeout(() => {
    if (SC.lastRec) document.getElementById('sc-status').textContent = 'applied';
  }, 2000);
}

// Wire pill clicks (delegation).
document.getElementById('sc-q1-pills').addEventListener('click', e => {
  const p = e.target.closest('.sc-pill');
  if (!p) return;
  SC.q1 = p.dataset.q1;
  scSelectPills('q1', SC.q1);
  // Default q2 to a sensible branch value if currently invalid for branch.
  if (SC.q1 === 'short' && !['right_now', 'session', 'next_session', 'multiday'].includes(SC.q2)) {
    SC.q2 = 'session';
  }
  if (SC.q1 === 'long' && !['solid', 'important', 'reference', 'mandatory'].includes(SC.q2)) {
    SC.q2 = 'solid';
  }
  scSelectPills('q2', SC.q2);
  scShowHide();
  SC.lastRec = null;
  document.getElementById('sc-status').textContent = 'press Recommend';
  document.getElementById('sc-apply-btn').disabled = true;
});
document.getElementById('sc-q2-short-pills').addEventListener('click', e => {
  const p = e.target.closest('.sc-pill');
  if (!p) return;
  SC.q2 = p.dataset.q2;
  scSelectPills('q2', SC.q2);
  SC.lastRec = null;
  document.getElementById('sc-status').textContent = 'press Recommend';
  document.getElementById('sc-apply-btn').disabled = true;
});
document.getElementById('sc-q2-long-pills').addEventListener('click', e => {
  const p = e.target.closest('.sc-pill');
  if (!p) return;
  SC.q2 = p.dataset.q2;
  scSelectPills('q2', SC.q2);
  SC.lastRec = null;
  document.getElementById('sc-status').textContent = 'press Recommend';
  document.getElementById('sc-apply-btn').disabled = true;
});

document.getElementById('sc-recommend-btn').addEventListener('click', scRecommend);
document.getElementById('sc-reset-btn').addEventListener('click', scReset);
document.getElementById('sc-apply-btn').addEventListener('click', scApply);

// Initial state.
scShowHide();
// Step 5 (B2, B7): restore persisted UI prefs (sort key, mandatory-first, search mode).
const LS_SORT_KEY = 'instinct:sortKey';
const LS_SORT_MAND = 'instinct:sortMandatoryFirst';
const LS_SEARCH_MODE = 'instinct:searchMode';
restoreSort();
restoreSearchMode();
// Wire the sort controls + search mode to persist on change.
document.getElementById('sort-key').addEventListener('change', () => { persistSort(); renderCurrentList(); });
document.getElementById('sort-mandatory-first').addEventListener('change', () => { persistSort(); renderCurrentList(); });
document.getElementById('search-mode').addEventListener('change', persistSearchMode);

async function loadStats() {
  try {
    const s = await api('/api/stats');
    // /api/stats returns total_segments (not count)
    const total = s.total_segments || s.count || 0;
    document.getElementById('stats').innerHTML =
      `<b>${total}</b> segments · store: <code>${escapeHtml(s.persist_dir || '')}</code>`;
  } catch (e) { toast(String(e), true); }
}

async function loadList() {
  try {
    const r = await api('/api/list?limit=500');
    allSegs = r.items;
    renderCurrentList();
  } catch (e) { toast(String(e), true); }
}

// ---- Step 5 (B7) — sort controls ----
// State persisted to localStorage so the user's preferred sort sticks
// across reloads. Also handles the "Mandatory first" checkbox which is
// applied as a secondary key on top of the primary sort.
function renderCurrentList() {
  const query = document.getElementById('search').value.toLowerCase();
  const mode = document.getElementById('search-mode').value;
  let items = allSegs;
  if (query && mode === 'filter') {
    items = allSegs.filter(segment => [segment.title, segment.content, segment.project]
      .some(value => (value || '').toLowerCase().includes(query)));
  } else if (query && mode === 'retriever' && lastRetrieverResult) {
    items = [...(lastRetrieverResult.mandatory || []), ...(lastRetrieverResult.relevant || [])];
  }
  applySort(items);
  renderList(items);
}

function applySort(items = allSegs) {
  const key = document.getElementById('sort-key').value;
  const mandatoryFirst = document.getElementById('sort-mandatory-first').checked;
  const popCount = s => (s.bump_count != null)
    ? s.bump_count
    : ((s.kpis && s.kpis.pop_count_total != null) ? s.kpis.pop_count_total : 0);
  const accessed = s => s.last_accessed || s.last_bumped_at || s.timestamp || '';
  items.sort((a, b) => {
    if (mandatoryFirst) {
      const a100 = (a.score === 100) ? 1 : 0;
      const b100 = (b.score === 100) ? 1 : 0;
      if (a100 !== b100) return b100 - a100;
    }
    switch (key) {
      case 'score-asc':       return a.score - b.score;
      case 'accessed-desc':   return accessed(b).localeCompare(accessed(a));
      case 'accessed-asc':    return accessed(a).localeCompare(accessed(b));
      case 'popped-desc':     return popCount(b) - popCount(a);
      case 'title-asc':       return (a.title || '').localeCompare(b.title || '');
      case 'score-desc':
      default:                return b.score - a.score;
    }
  });
}

function persistSort() {
  try {
    localStorage.setItem(LS_SORT_KEY, document.getElementById('sort-key').value);
    localStorage.setItem(LS_SORT_MAND, document.getElementById('sort-mandatory-first').checked ? '1' : '0');
  } catch (_) { /* localStorage may be disabled — non-fatal */ }
}

function restoreSort() {
  try {
    const k = localStorage.getItem(LS_SORT_KEY);
    const m = localStorage.getItem(LS_SORT_MAND);
    if (k) document.getElementById('sort-key').value = k;
    if (m != null) document.getElementById('sort-mandatory-first').checked = m === '1';
  } catch (_) { /* ignore */ }
}

// ---- Step 5 (B2) — search-mode persistence ----
// Persist the search-mode dropdown (filter vs retriever) across reloads.
function persistSearchMode() {
  try {
    localStorage.setItem(LS_SEARCH_MODE, document.getElementById('search-mode').value);
  } catch (_) { /* ignore */ }
}
function restoreSearchMode() {
  try {
    const m = localStorage.getItem(LS_SEARCH_MODE);
    if (m === 'filter' || m === 'retriever') {
      document.getElementById('search-mode').value = m;
    }
  } catch (_) { /* ignore */ }
}

// ---- Step 5 (B3) — bulk-action bar ----
// State is local to the page (not persisted) — selection is reset on
// reload by design, since segment IDs may not survive a session restart.
function onBulkToggle() {
  applyBulkState();
}

function bulkSelectedIds() {
  const cbs = document.querySelectorAll('input.bulk-cb:checked');
  return Array.from(cbs).map(c => c.dataset.id);
}

function applyBulkState() {
  const ids = bulkSelectedIds();
  const bar = document.getElementById('bulk-bar');
  // Mirror the count to both the bar row and the section-header pill.
  const rowCount = document.getElementById('bulk-count-row');
  const pillCount = document.getElementById('bulk-count-pill');
  const b = document.getElementById('bulk-count');
  if (b) b.textContent = ids.length;
  if (rowCount) rowCount.textContent = ids.length;
  if (pillCount) pillCount.style.display = ids.length > 0 ? 'inline' : 'none';
  bar.classList.toggle('show', ids.length > 0);
}

function bulkClear() {
  document.querySelectorAll('input.bulk-cb').forEach(c => { c.checked = false; });
  applyBulkState();
}

async function bulkBumpScore(delta) {
  const ids = bulkSelectedIds();
  if (!ids.length) return;
  if (!confirm(`Bump score by ${delta} on ${ids.length} segment(s)?`)) return;
  let ok = 0, fail = 0;
  for (const id of ids) {
    const seg = allSegs.find(s => s.id === id);
    if (!seg) { fail++; continue; }
    const newScore = Math.max(0, Math.min(100, seg.score + delta));
    try {
      await api('/api/update_score', {
        method: 'POST',
        body: { id, score: newScore, reason: `bulk ${delta > 0 ? '+' : ''}${delta}` },
      });
      seg.score = newScore;
      ok++;
    } catch (e) { fail++; }
  }
  toast(`Bulk ${delta > 0 ? '+' : ''}${delta}: ${ok} ok, ${fail} failed`);
  if (delta === 0 && !ids.length) toast('No segments selected', true);
  bulkClear();
  applySort();
  renderList(allSegs);
}

async function bulkSetScore() {
  const ids = bulkSelectedIds();
  if (!ids.length) return;
  const raw = prompt(`Set absolute score on ${ids.length} segment(s) (0-100):`);
  if (raw == null) return;
  const target = Number(raw);
  if (!raw.trim() || !Number.isInteger(target) || target < 0 || target > 100) {
    toast('Score must be 0-100', true); return;
  }
  let ok = 0, fail = 0;
  for (const id of ids) {
    try {
      await api('/api/update_score', { method: 'POST', body: { id, score: target, reason: 'bulk set' } });
      const seg = allSegs.find(s => s.id === id);
      if (seg) seg.score = target;
      ok++;
    } catch (e) { fail++; }
  }
  toast(`Bulk set ${target}: ${ok} ok, ${fail} failed`);
  bulkClear();
  applySort();
  renderList(allSegs);
}

async function bulkRecycle() {
  const ids = bulkSelectedIds();
  if (!ids.length) return;
  if (!confirm(`Move ${ids.length} segment(s) to the recycle bin?`)) return;
  let ok = 0, fail = 0;
  for (const id of ids) {
    try {
      await api('/api/recycle', { method: 'POST', body: { id } });
      ok++;
    } catch (e) { fail++; }
  }
  toast(`Bulk recycle: ${ok} ok, ${fail} failed`);
  bulkClear();
  await loadList();
}

async function pullMandatory() {
  // One-click: simulate the agent retrieving every mandatory
  // (score-99-100) segment. This actually calls record_access via
  // /api/access, which triggers the nudge drop (99/100->79) — the same
  // path the MCP retriever uses. After this returns, the list is
  // refreshed and the dropped segments disappear from the top of
  // the list (they're now 79, not 100). Useful for confirming that
  // the nudge system is wired correctly and for testing the cycle.
  if (!allSegs.length) { toast('List not loaded yet', true); return; }
  const mandatory = allSegs.filter(s => s.score >= 99);
  if (!mandatory.length) { toast('No mandatory segments at 99+', true); return; }
  try {
    const r = await api('/api/access', {
      method: 'POST',
      body: { ids: mandatory.map(s => s.id) },
    });
    toast(`Accessed ${r.recorded} mandatory segment(s) — they dropped to 79. Will re-elevate after the nudge interval.`);
  } catch (e) {
    toast('Access failed: ' + e.message, true);
    return;
  }
  // Scroll to the top (the dropped ones are no longer at 99+)
  const list = document.getElementById('seg-list');
  list.scrollTop = 0;
  // Reload so the list reflects the new scores
  await loadList();
}

async function selectSeg(id) {
  const s = await api('/api/get?id=' + encodeURIComponent(id));
  document.getElementById('f-id').value = s.id;
  document.getElementById('f-title').value = s.title || '';
  document.getElementById('f-project').value = s.project || '';
  document.getElementById('f-score').value = s.score;
  document.getElementById('f-type').value = s.segment_type || 'context';
  document.getElementById('f-tags').value = (s.tags || []).join(', ');
  document.getElementById('f-profile').value = s.decay_profile || 'standard';
  document.getElementById('f-bump-count').value = s.bump_count || 0;
  document.getElementById('f-last-bump').value = s.last_bumped_at || 'never';
  document.getElementById('f-content').value = s.content || '';
  document.getElementById('del-btn').style.display   = s.recycled ? 'none' : '';
  document.getElementById('purge-btn').style.display  = s.recycled ? '' : 'none';
  document.getElementById('restore-btn').style.display = s.recycled ? '' : 'none';
  // Cache the selected id globally so onRecycle/onRestore/onPurge know what to act on
  window._selectedId = s.id;
  window._selectedSeg = s;
  document.getElementById('seg-detail').textContent = JSON.stringify(s, null, 2);
  document.getElementById('form-title').textContent = 'Edit segment';
  const nudge = s.nudge || {};
  document.getElementById('nudge-controls').style.display = nudge.is_mandatory_type ? '' : 'none';
  document.getElementById('nudge-interval').value = nudge.nudge_interval_turns ?? 0;
  document.getElementById('nudge-active').checked = !!nudge.nudge_active;
  dashboardApplyWriteState();
  document.querySelectorAll('.seg-item').forEach(el => {
    el.classList.toggle('selected', el.dataset.id === id);
  });
  // Step 7 (C2) — show diff between stored and current edit-form state.
  renderSegmentDiff();
}

function resetForm() {
  document.getElementById('seg-form').reset();
  document.getElementById('f-id').value = '';
  document.getElementById('f-score').value = 50;
  document.getElementById('f-profile').value = 'standard';
  document.getElementById('f-bump-count').value = '';
  document.getElementById('f-last-bump').value = '';
  document.getElementById('form-title').textContent = 'Add segment';
  document.getElementById('del-btn').style.display   = 'none';
  document.getElementById('purge-btn').style.display  = 'none';
  document.getElementById('restore-btn').style.display = 'none';
  window._selectedId = null;
  window._selectedSeg = null;
  document.getElementById('nudge-controls').style.display = 'none';
  document.getElementById('seg-detail').textContent = '(no segment selected)';
  document.querySelectorAll('.seg-item.selected').forEach(el => el.classList.remove('selected'));
  // Step 7 (C2) — clear the diff panel.
  const d = document.getElementById('seg-diff');
  if (d) d.innerHTML = '<div class="diff-empty">Edit the form above to see live diffs vs. the stored segment.</div>';
}

// Step 7 (C2) — Diff rendering. Compares the current edit-form values
// against the stored segment. Renders one row per field showing whether
// it changed. Cheap: works entirely client-side, updates on input.
const _DIFF_FIELDS = [
  { id: 'f-title',    storedKey: 'title',        label: 'title',    kind: 'text' },
  { id: 'f-project',  storedKey: 'project',      label: 'project',  kind: 'text' },
  { id: 'f-score',    storedKey: 'score',        label: 'score',    kind: 'int'  },
  { id: 'f-type',     storedKey: 'segment_type', label: 'type',     kind: 'text' },
  { id: 'f-tags',     storedKey: 'tags',         label: 'tags',     kind: 'tags' },
  { id: 'f-profile',  storedKey: 'decay_profile',label: 'profile',  kind: 'text' },
  { id: 'f-content',  storedKey: 'content',      label: 'content',  kind: 'text' },
];

function _formValue(field, el) {
  const raw = el.value;
  if (field.kind === 'int') return parseInt(raw, 10);
  if (field.kind === 'tags') {
    return raw.split(',').map(s => s.trim()).filter(Boolean);
  }
  return raw;
}

function _storedValue(field, s) {
  if (s == null) return undefined;
  if (field.kind === 'tags') return s[field.storedKey] || [];
  if (field.kind === 'int')   return Number(s[field.storedKey] ?? 0);
  return s[field.storedKey] ?? '';
}

function _eq(a, b) {
  if (Array.isArray(a) && Array.isArray(b)) {
    return a.length === b.length && a.every((v, i) => v === b[i]);
  }
  return String(a) === String(b);
}

function renderSegmentDiff() {
  const d = document.getElementById('seg-diff');
  if (!d) return;
  if (!window._selectedSeg) {
    d.innerHTML = '<div class="diff-empty">Edit the form above to see live diffs vs. the stored segment.</div>';
    return;
  }
  const s = window._selectedSeg;
  let html = '';
  let anyChange = false;
  for (const f of _DIFF_FIELDS) {
    const el = document.getElementById(f.id);
    if (!el) continue;
    const cur = _formValue(f, el);
    const sto = _storedValue(f, s);
    const same = _eq(cur, sto);
    if (same) {
      html += `<span class="diff-line diff-same"><span class="diff-field">${f.label}:</span> (unchanged) ${_shortVal(cur)}</span>`;
    } else {
      anyChange = true;
      html += `<span class="diff-line diff-changed"><span class="diff-field">${f.label}:</span></span>`;
      html += `<span class="diff-line diff-removed">  - ${_shortVal(sto)}</span>`;
      html += `<span class="diff-line diff-added">  + ${_shortVal(cur)}</span>`;
    }
  }
  if (!anyChange) {
    html = '<div class="diff-empty">No pending changes — form matches stored segment.</div>' + html;
  } else {
    html = '<div class="diff-empty" style="color:#fc6">⚠ Pending changes — Save to apply, or select another segment to discard.</div>' + html;
  }
  d.innerHTML = html;
}

function _shortVal(v) {
  if (v === undefined || v === null) return '(empty)';
  if (Array.isArray(v)) return v.length ? escapeHtml('[' + v.join(', ') + ']') : '(empty)';
  const s = String(v);
  if (s.length > 80) return escapeHtml(s.slice(0, 77) + '…');
  return escapeHtml(s);
}

// Re-render the diff whenever any form field changes.
for (const f of _DIFF_FIELDS) {
  const el = document.getElementById(f.id);
  if (el) el.addEventListener('input', renderSegmentDiff);
}

async function onSubmit(ev) {
  ev.preventDefault();
  const id = document.getElementById('f-id').value;
  const payload = {
    title: document.getElementById('f-title').value,
    content: document.getElementById('f-content').value,
    score: Number(document.getElementById('f-score').value),
    segment_type: document.getElementById('f-type').value,
    project: document.getElementById('f-project').value,
    tags: document.getElementById('f-tags').value,
    decay_profile: document.getElementById('f-profile').value,
  };
  try {
    if (id) {
      // update path: score + profile (re-add would lose the id)
      await api('/api/update', { method: 'POST', body: {id, ...payload} });
      toast('Updated');
      await selectSeg(id);
    } else {
      const r = await api('/api/add', { method: 'POST', body: payload });
      toast('Added ' + r.id.slice(0, 8));
    }
    await loadList(); await loadStats();
  } catch (e) { toast(String(e), true); }
  return false;
}

async function onRecycle() {
  // Soft delete: hides the segment from list/search/retriever but
  // keeps it in the recycle bin. Default delete action in the UI.
  // Step 3 A6 — replace plain toast with an undo toast.
  if (!window._selectedId) { toast('Select a segment first', true); return; }
  const seg = window._selectedSeg;
  const id = window._selectedId;
  const titleShort = seg && seg.title ? seg.title : id.slice(0, 8);
  if (!confirm('♻ Recycle "' + titleShort + '"?\n\nIt will be hidden from search and list, but can be restored from the Recycle bin.')) return;
  const reason = prompt('Optional reason:', '') || '';
  try {
    await api('/api/recycle', { method: 'POST', body: { id, by: 'dashboard', reason, title: seg ? seg.title : '' } });
    await loadList(); await loadStats();
    resetForm();
    // Show sticky toast with Undo button. The Undo click restores
    // the just-recycled segment.
    toast('♻ Recycled "' + titleShort + '"', {
      duration: 8000,
      actions: [{
        label: 'Undo',
        onClick: async () => {
          try {
            await api('/api/restore', { method: 'POST', body: { id } });
            toast('↺ Restored "' + titleShort + '"', { success: true });
            await loadList(); await loadStats();
          } catch (e) {
            toast('Undo failed: ' + String(e), true);
          }
        },
      }],
    });
  } catch (e) { toast(String(e), true); }
}

async function onRestore() {
  // Undo a recycle. Brings the segment back to the active set.
  if (!window._selectedId) { toast('Select a segment first', true); return; }
  const id = window._selectedId;
  const titleShort = window._selectedSeg ? window._selectedSeg.title : id.slice(0, 8);
  try {
    await api('/api/restore', { method: 'POST', body: { id } });
    toast('↺ Restored "' + titleShort + '"', { success: true });
    await loadList(); await loadStats();
    resetForm();
  } catch (e) { toast(String(e), true); }
}

async function onPurge() {
  // HARD delete. IRREVERSIBLE. Only available when the segment is
  // already in the recycle bin. Requires explicit "I AM SURE" prompt.
  if (!window._selectedId) { toast('Select a segment first', true); return; }
  if (!confirm('✖ PURGE (HARD DELETE)?\n\nThis is IRREVERSIBLE. The segment will be permanently destroyed.\n\nType the word PURGE in the next prompt to confirm.')) return;
  const word = prompt('Type PURGE to confirm:');
  if (word !== 'PURGE') { toast('Cancelled (typed: ' + (word || 'nothing') + ')', true); return; }
  try {
    await api('/api/purge', { method: 'POST', body: { id: window._selectedId } });
    toast('✖ Purged (hard-deleted)', { success: true });
    await loadList(); await loadStats();
    resetForm();
    // If the recycle modal is open, refresh it so the purged row disappears.
    if (document.getElementById('rb-overlay').classList.contains('open')) {
      await rbRefresh();
    }
  } catch (e) { toast(String(e), true); }
}

// ---- Step 3 A3 — Recycle-bin modal ----
async function loadRecycled() {
  // Open the recycle-bin modal. Fetch fresh, render rows, wire per-row
  // actions + bulk actions (restore-all, empty-bin).
  document.getElementById('rb-overlay').classList.add('open');
  await rbRefresh();
}

async function rbRefresh() {
  const body = document.getElementById('rb-body');
  const count = document.getElementById('rb-count');
  try {
    const r = await api('/api/recycled');
    const items = r.items || [];
    count.textContent = '(' + items.length + (items.length === 1 ? ' segment' : ' segments') + ')';
    document.getElementById('rb-restore-all').disabled = items.length === 0;
    document.getElementById('rb-empty-bin').disabled = items.length === 0;
    if (items.length === 0) {
      body.innerHTML = '<div class="rb-empty">Recycle bin is empty. ♻</div>';
      return;
    }
      body.innerHTML = items.map(it => rbRowHtml(it)).join('');
      dashboardApplyWriteState();
    // Wire per-row buttons.
    body.querySelectorAll('button[data-action]').forEach(btn => {
      btn.addEventListener('click', () => {
        const id = btn.dataset.id;
        const action = btn.dataset.action;
        if (action === 'restore') rbRestoreOne(id);
        else if (action === 'purge') rbPurgeOne(id);
      });
    });
  } catch (e) {
    body.innerHTML = '<div class="rb-empty">⚠ ' + escapeHtml(String(e)) + '</div>';
  }
}

function rbRowHtml(it) {
  const id = it.id || '';
  const idShort = id.slice(0, 8);
  const title = escapeHtml(it.title || '(no title)');
  const score = it.score != null ? it.score : '-';
  const segType = escapeHtml(it.segment_type || '-');
  const md = it;
  const recycledAt = escapeHtml(md.recycled_at || '?');
  const recycledBy = escapeHtml(md.recycled_by || '?');
  const reason = md.recycled_reason ? escapeHtml(md.recycled_reason) : '';
  return `
    <div class="rb-row" data-id="${escapeHtml(id)}">
      <div class="rb-title">${title}<small>id: ${idShort}${reason ? ' · reason: ' + reason : ''}</small></div>
      <div class="rb-meta"><span class="score${score >= 95 ? ' m' : (score >= 80 ? ' h' : '')}">${score}</span></div>
      <div class="rb-meta">${segType}</div>
      <div class="rb-meta" title="recycled_at">${recycledAt}</div>
      <div class="rb-meta" title="recycled_by">by ${recycledBy}</div>
      <div class="rb-actions">
        <button class="restore" data-action="restore" data-id="${escapeHtml(id)}" title="Restore back to active set">↺ Restore</button>
        <button class="purge" data-action="purge" data-id="${escapeHtml(id)}" title="HARD delete — irreversible">✖ Purge</button>
      </div>
    </div>`;
}

async function rbRestoreOne(id) {
  try {
    await api('/api/restore', { method: 'POST', body: { id } });
    toast('↺ Restored', { success: true });
    await loadList(); await loadStats();
    await rbRefresh();
  } catch (e) { toast(String(e), true); }
}

async function rbPurgeOne(id) {
  if (!confirm('✖ PURGE this segment? IRREVERSIBLE.')) return;
  if (prompt('Type PURGE to confirm:') !== 'PURGE') { toast('Cancelled', true); return; }
  try {
    await api('/api/purge', { method: 'POST', body: { id } });
    toast('✖ Purged', { success: true });
    await loadList(); await loadStats();
    await rbRefresh();
  } catch (e) { toast(String(e), true); }
}

async function rbRestoreAll() {
  const body = document.getElementById('rb-body');
  const count = (body.querySelectorAll('.rb-row').length);
  if (count === 0) return;
  if (!confirm(`↺ Restore all ${count} recycled segments back to the active set?`)) return;
  try {
    const r = await api('/api/recycled');
    let ok = 0, fail = 0;
    for (const it of r.items) {
      try {
        await api('/api/restore', { method: 'POST', body: { id: it.id } });
        ok++;
      } catch { fail++; }
    }
    toast(`Restored ${ok}${fail ? `, ${fail} failed` : ''}`, { success: fail === 0, duration: 4000 });
    await loadList(); await loadStats();
    await rbRefresh();
  } catch (e) { toast(String(e), true); }
}

async function rbEmptyBin() {
  const body = document.getElementById('rb-body');
  const count = (body.querySelectorAll('.rb-row').length);
  if (count === 0) return;
  if (!confirm(`✖ EMPTY BIN — hard-delete ${count} segments?\n\nIRREVERSIBLE. These will be PERMANENTLY DESTROYED.`)) return;
  if (prompt('Type EMPTY to confirm:') !== 'EMPTY') { toast('Cancelled', true); return; }
  try {
    const r = await api('/api/recycled');
    let ok = 0, fail = 0;
    for (const it of r.items) {
      try {
        await api('/api/purge', { method: 'POST', body: { id: it.id } });
        ok++;
      } catch { fail++; }
    }
    toast(`Emptied bin (${ok} purged${fail ? `, ${fail} failed` : ''})`, { success: fail === 0, duration: 4000 });
    await loadList(); await loadStats();
    await rbRefresh();
  } catch (e) { toast(String(e), true); }
}

// Wire modal close + bulk buttons + Esc.
document.getElementById('rb-close').addEventListener('click', () => {
  document.getElementById('rb-overlay').classList.remove('open');
});
document.getElementById('rb-restore-all').addEventListener('click', rbRestoreAll);
document.getElementById('rb-empty-bin').addEventListener('click', rbEmptyBin);
document.getElementById('rb-overlay').addEventListener('click', (ev) => {
  if (ev.target === ev.currentTarget) ev.currentTarget.classList.remove('open');
});
document.addEventListener('keydown', (ev) => {
  if (ev.key === 'Escape') {
    const ov = document.getElementById('rb-overlay');
    if (ov.classList.contains('open')) ov.classList.remove('open');
    const kb = document.getElementById('kb-overlay');
    if (kb.style.display !== 'none') kb.style.display = 'none';
  }
});

// Step 7 (C1) — Keyboard shortcuts.
// - j / ArrowDown  : select next visible segment
// - k / ArrowUp    : select previous visible segment
// - /              : focus the search box
// - ?              : open the help overlay
// - n              : focus the new-segment form (clears selection)
// - e              : focus the edit form (when a segment is selected)
// - r              : recycle the selected segment
// - t              : toggle light / dark theme  (C4 hook — defined later)
// - x              : export JSON  (C5 hook — defined later)
// - Esc            : close help / modal / unfocus search (above)
function isTypingTarget(el) {
  if (!el) return false;
  const tag = (el.tagName || '').toUpperCase();
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return true;
  if (el.isContentEditable) return true;
  return false;
}

function navigateSegment(direction) {
  // direction = +1 (next) or -1 (prev)
  const items = [...document.querySelectorAll('#seg-list .seg-item')];
  if (!items.length) return;
  let idx = items.findIndex(el => el.dataset.id === window._selectedId);
  if (idx < 0) idx = direction > 0 ? -1 : items.length;
  idx = (idx + direction + items.length) % items.length;
  const target = items[idx];
  if (target) {
    selectSeg(target.dataset.id);
    target.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }
}

function openHelp() {
  const kb = document.getElementById('kb-overlay');
  kb.style.display = 'flex';
}

function focusSearch() {
  const s = document.getElementById('search');
  if (s) { s.focus(); s.select(); }
}

function focusNewForm() {
  resetForm();
  document.getElementById('f-title').focus();
}

function focusEditForm() {
  if (!window._selectedId) {
    toast('No segment selected. Press n to start a new one, or j/k to navigate.', { duration: 2500 });
    return;
  }
  document.getElementById('f-title').focus();
  document.getElementById('f-title').select();
}

function recycleSelected() {
  if (!window._selectedId) {
    toast('Select a segment first (j/k or click)', true);
    return;
  }
  if (confirm(`Recycle segment "${window._selectedSeg?.title || window._selectedId}"?`)) {
    onRecycle();
  }
}

document.addEventListener('keydown', (ev) => {
  // Esc always works (handled above).
  if (ev.key === 'Escape') return;
  // Skip when the user is typing into a text field.
  if (isTypingTarget(ev.target)) return;
  // Skip when modifier keys are held (don't hijack browser shortcuts).
  if (ev.ctrlKey || ev.metaKey || ev.altKey) return;

  const key = ev.key;
  if (key === '?') { openHelp(); ev.preventDefault(); return; }
  if (key === '/') { focusSearch(); ev.preventDefault(); return; }
  if (key === 'j' || key === 'ArrowDown') { navigateSegment(+1); ev.preventDefault(); return; }
  if (key === 'k' || key === 'ArrowUp')   { navigateSegment(-1); ev.preventDefault(); return; }
  if (key === 'n') { focusNewForm(); ev.preventDefault(); return; }
  if (key === 'e') { focusEditForm(); ev.preventDefault(); return; }
  if (key === 'r') { recycleSelected(); ev.preventDefault(); return; }
  if (key === 't') { if (typeof toggleTheme === 'function') toggleTheme(); ev.preventDefault(); return; }
  if (key === 'x') { if (typeof exportJson === 'function') exportJson(); ev.preventDefault(); return; }
});

// Close-button wiring for the help overlay.
document.getElementById('kb-close').addEventListener('click', () => {
  document.getElementById('kb-overlay').style.display = 'none';
});
document.getElementById('kb-overlay').addEventListener('click', (ev) => {
  if (ev.target === ev.currentTarget) ev.currentTarget.style.display = 'none';
});

// Step 7 (C3) — Read-only preference.
// Client-side state is synchronized with the SQLite observer/owner role.
async function saveNudge() {
  try {
    const id = document.getElementById('f-id').value;
    if (!id) throw new Error('Select a mandatory memory first');
    const turns = Number(document.getElementById('nudge-interval').value);
    if (!Number.isInteger(turns) || turns < 0) throw new Error('Turns must be a nonnegative integer');
    await api('/api/nudge/interval', {method:'POST', body:{id, turns, active:document.getElementById('nudge-active').checked}});
    await selectSeg(id); toast('Nudge settings saved');
  } catch (error) { toast(String(error), true); }
}
async function runNudge() {
  try { const result = await api('/api/nudge/run', {method:'POST', body:{}}); toast(JSON.stringify(result)); await loadList(); }
  catch (error) { toast(String(error), true); }
}
// Step 7 (C4) — Light/dark theme toggle.
// Preference is stored in localStorage under 'instinct:theme'. Default is
// 'dark' (matches the current palette). Toggling flips the body[data-theme]
// attribute; CSS overrides take care of the rest.
const LS_THEME = 'instinct:theme';
function applyTheme(theme) {
  if (theme === 'light') {
    document.body.setAttribute('data-theme', 'light');
  } else {
    document.body.removeAttribute('data-theme');
  }
  const btn = document.getElementById('theme-toggle');
  if (btn) btn.textContent = theme === 'light' ? '🌙' : '☀';
}
function toggleTheme() {
  const cur = document.body.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
  const next = cur === 'light' ? 'dark' : 'light';
  try { localStorage.setItem(LS_THEME, next); } catch {}
  applyTheme(next);
  toast(`Theme: ${next}`, { duration: 1200 });
}
// Restore theme on load.
(function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem(LS_THEME); } catch {}
  applyTheme(saved === 'light' ? 'light' : 'dark');
})();

// Step 7 (C3) — restore the read-only preference on load.

// Step 7 (C5) — JSON export. Downloads all segments (active + recycled) as
// a single .json file the user can keep or feed back into the store later.
async function exportJson() {
  try {
    const [active, recycled] = await Promise.all([
      api('/api/list?limit=1000'),
      api('/api/recycled'),
    ]);
    const payload = {
      schema: 'instinct-dashboard-export-v1',
      exported_at: new Date().toISOString(),
      complete: active.count === (active.items || []).length + (recycled.items || []).length && (recycled.items || []).length < 500,
      limits: {active:1000, recycled:500},
      counts: {
        active: (active.items || []).length,
        recycled: (recycled.items || []).length,
      },
      active: active.items || [],
      recycled: recycled.items || [],
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    a.download = `instinct-export-${stamp}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    toast(`Exported ${payload.counts.active} active + ${payload.counts.recycled} recycled${payload.complete ? '' : ' — incomplete bounded snapshot, not a backup'}`, { success: payload.complete, error: !payload.complete });
  } catch (e) {
    toast('Export failed: ' + (e && e.message || e), true);
  }
}

// Track the most recent retriever result so the "Pop audit" button can
// re-open it (and so we don't re-run the search on every pop-audit click).
let lastRetrieverResult = null;

// First-time hint: surface the search-mode control when the user focuses
// the search box for the first time per page load. Dismissed by Esc or
// by changing the mode.
let searchHintDismissed = false;
function maybeShowSearchHint() {
  if (searchHintDismissed) return;
  const banner = document.getElementById('search-banner');
  const mode = document.getElementById('search-mode').value;
  if (mode === 'retriever') {
    banner.style.background = '#2a2a3a';
    banner.style.display = 'block';
    banner.innerHTML = `<b>Retriever mode active</b> — typing runs the full agent-style retriever (mandatory + relevant + pop rules) with a 300ms debounce. Press <b>Enter</b> to force-run, <b>Esc</b> to clear. Switch the dropdown to "Local filter" for instant substring match.`;
  } else {
    banner.style.background = '#2a3a2a';
    banner.style.display = 'block';
    banner.innerHTML = `<b>Local filter mode</b> — typing does a fast substring match over the loaded list (no retriever, no pop rules). Switch the dropdown to "Retriever search" to run the full agent-style retriever with pop rules.`;
  }
}
document.getElementById('search').addEventListener('focus', maybeShowSearchHint);
document.getElementById('search').addEventListener('blur', () => {
  // Auto-hide the hint after a short delay (so the click can register)
  setTimeout(() => {
    if (!document.getElementById('search').matches(':focus')) {
      // Keep the banner up only if there are pop results to show
      if (!lastRetrieverResult || !lastRetrieverResult.pop_count) {
        document.getElementById('search-banner').style.display = 'none';
      }
    }
  }, 200);
});

document.getElementById('search').addEventListener('input', (e) => {
  const mode = document.getElementById('search-mode').value;
  const q = e.target.value;
  if (!q) {
    document.getElementById('search-banner').style.display = 'none';
    return renderList(allSegs);
  }
  if (mode === 'filter') {
    const lower = q.toLowerCase();
    return renderList(allSegs.filter(s =>
      (s.title || '').toLowerCase().includes(lower) ||
      (s.content || '').toLowerCase().includes(lower) ||
      (s.project || '').toLowerCase().includes(lower)
    ));
  }
  // Retriever mode: type-to-search with a 300ms debounce so we don't
  // hammer the retriever on every keystroke.
  clearTimeout(window.__searchDebounce);
  window.__searchDebounce = setTimeout(() => retrieverSearch(q), 300);
});

document.getElementById('search').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    e.preventDefault();
    clearTimeout(window.__searchDebounce);
    const mode = document.getElementById('search-mode').value;
    const q = e.target.value.trim();
    if (!q) return;
    if (mode === 'retriever') {
      retrieverSearch(q);
    } else {
      // Enter in filter mode does nothing extra; the 'input' handler already ran.
    }
  } else if (e.key === 'Escape') {
    e.target.value = '';
    e.target.blur();
    searchHintDismissed = true;
    document.getElementById('search-banner').style.display = 'none';
    renderList(allSegs);
  }
});

document.getElementById('search-mode').addEventListener('change', (e) => {
  searchHintDismissed = true;  // user has interacted with the mode control
  const q = document.getElementById('search').value;
  if (!q) {
    document.getElementById('search-banner').style.display = 'none';
    return;
  }
  if (e.target.value === 'retriever') {
    retrieverSearch(q);
  } else {
    const lower = q.toLowerCase();
    renderList(allSegs.filter(s =>
      (s.title || '').toLowerCase().includes(lower) ||
      (s.content || '').toLowerCase().includes(lower) ||
      (s.project || '').toLowerCase().includes(lower)
    ));
  }
});

async function retrieverSearch(q) {
  // Full retriever pipeline: same path the agent uses to get context.
  try {
    const r = await dashboardFetch('/api/search_with_pop', {
      method: 'POST', body: JSON.stringify({q, n: 20, apply_pops: instinctCanWrite(), track_access: false})
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const d = await r.json();
    lastRetrieverResult = d;
    renderRetrieverResult(d);
  } catch (e) {
    toast('Retriever search failed: ' + e, true);
  }
}

function renderRetrieverResult(d) {
  // Build the combined list: mandatory first (always surfaced), then
  // relevant (decay-filtered, pop-mutated). This is what the agent sees.
  const items = [...(d.mandatory || []), ...(d.relevant || [])];
  renderList(items);

  // Pop audit banner (compact, inline). The "Pop audit" button opens
  // a detailed modal of the same data.
  const banner = document.getElementById('search-banner');
  if (!d.pop_count) {
    banner.innerHTML = `<b>Retriever result</b>: mandatory=${d.mandatory_count}, relevant=${d.relevant_count}, pops=0 (no keyword matches ≥2)`;
    banner.style.background = '#1a2a1a';
  } else {
    const pops = (d.pop_results || []).map(p => {
      const ruleColor = ({
        'mandatory_bump': '#fc6', 'halve': 'var(--brand-bright)', 'jump_to_plateau': '#f96',
        'frequency_reset': '#f66', 'none': '#888',
      })[p.rule_applied] || '#fff';
      return `<span style="color:${ruleColor}">${escapeHtml(p.rule_applied)}</span> on <b>${escapeHtml((p.segment_title || '').slice(0, 40))}</b> (matched=${escapeHtml(p.matched_keywords)})`;
    }).join('; ');
    banner.innerHTML = `<b>Retriever result</b>: mandatory=${d.mandatory_count}, relevant=${d.relevant_count}, <b style="color:var(--brand-bright)">pops=${d.pop_count}</b> — ${pops} &nbsp; <a href="#" onclick="showPopAudit(lastRetrieverResult); return false" style="color:var(--brand-bright)">details →</a>`;
    banner.style.background = '#2a3a2a';
  }
  banner.style.display = 'block';
}

async function onPopSearch() {
  const q = document.getElementById('search').value.trim();
  if (!q) { toast('Enter a search term first', true); return; }
  if (lastRetrieverResult && lastRetrieverResult.query === q) {
    return showPopAudit(lastRetrieverResult);
  }
  // No prior result for this query — run the retriever first.
  const btn = document.getElementById('pop-search-btn');
  const oldText = btn.textContent;
  btn.disabled = true; btn.textContent = '⚡ Running...';
  try {
    const r = await dashboardFetch('/api/search_with_pop', {
      method: 'POST', body: JSON.stringify({q, n: 10, apply_pops: instinctCanWrite(), track_access: false})
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const d = await r.json();
    lastRetrieverResult = d;
    showPopAudit(d);
  } catch (e) {
    toast('Pop audit failed: ' + e, true);
  } finally {
    btn.disabled = false; btn.textContent = oldText;
  }
}

function showPopAudit(d) {
  // Build a small modal showing what popped (or "no pops fired")
  const overlay = document.createElement('div');
  overlay.id = 'pop-audit-overlay';
  overlay.setAttribute('role', 'dialog');
  overlay.tabIndex = -1;
  overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.7);z-index:1000;display:flex;align-items:center;justify-content:center;padding:20px';
  const card = document.createElement('div');
  card.style.cssText = 'background:#1e1e1e;color:#ddd;border:1px solid #555;border-radius:6px;padding:18px;max-width:780px;max-height:80vh;overflow:auto;font-family:monospace;font-size:13px';

  const header = `<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
    <b style="color:var(--brand-bright)">Pop audit</b>
    <button onclick="document.getElementById('pop-audit-overlay').remove()" style="background:#444;color:#ddd;border:1px solid #666;padding:4px 10px;cursor:pointer">×</button>
  </div>`;

  const summary = `<div style="margin-bottom:10px;padding:8px;background:#222;border-radius:4px">
    <b>Query:</b> <code>${(d.query || '').replace(/[<>&]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;'})[c])}</code><br>
    <b>Mandatory:</b> ${d.mandatory_count} &nbsp; <b>Relevant:</b> ${d.relevant_count} &nbsp; <b>Pops fired:</b> <span style="color:${d.pop_count > 0 ? 'var(--brand-bright)' : '#888'}">${d.pop_count}</span>
  </div>`;

  const popList = (d.pop_results || []).map(p => {
    const ruleColor = ({
      'mandatory_bump': '#fc6',
      'halve': 'var(--brand-bright)',
      'jump_to_plateau': '#f96',
      'frequency_reset': '#f66',
      'none': '#888',
    })[p.rule_applied] || '#fff';
    return `<div style="margin:6px 0;padding:8px;background:#2a2a2a;border-left:3px solid ${ruleColor};border-radius:3px">
      <div><b>${escapeHtml(p.segment_title)}</b>${p.was_mandatory ? ' <span style="color:#fc6">[mandatory]</span>' : ''}</div>
      <div style="margin-top:4px;color:#aaa">rule=<span style="color:${ruleColor}">${escapeHtml(p.rule_applied)}</span> &nbsp; matched=${escapeHtml(p.matched_keywords)} keywords</div>
      <div style="color:#aaa">initial_score: ${p.initial_score_before} → ${p.initial_score_after}</div>
      <div style="color:#aaa">effective_score: ${p.effective_score_before} → ${p.effective_score_after}</div>
      <div style="color:#888;font-size:11px;margin-top:2px">${escapeHtml(p.message)}</div>
    </div>`;
  }).join('');

  const noPops = d.pop_count === 0
    ? '<div style="color:#888;padding:8px">No pops fired. (Need ≥2 keyword matches in segment identifiers; the bar is at 2-keyword halves, 3-keyword jumps, and frequency on 2nd pop in 8h.)</div>'
    : '';

  card.innerHTML = header + summary + (popList || noPops);
  overlay.appendChild(card);
  overlay.addEventListener('click', e => { if (e.target === overlay) overlay.remove(); });
  document.body.appendChild(overlay);
  overlay.addEventListener('keydown', event => { if (event.key === 'Escape') overlay.remove(); });
  overlay.focus();
}

(async function init() {
  // Check read-only state from /api/health. If the dashboard is in
  // read-only mode (MCP server or another process holds the writer
  // lock), show the banner at the top.
  try {
    const h = await api('/api/health');
    if (h.read_only) {
      document.getElementById('readonly-banner').style.display = 'block';
      // Disable the write buttons visually so the user knows without
      // trying. The server will also reject with 409.
      const ids = ['save-btn', 'del-btn'];
      ids.forEach(id => {
        const el = document.getElementById(id);
        if (el) { el.disabled = true; el.title = 'read-only SQLite observer'; }
      });
    }
  } catch (e) { /* non-fatal */ }
  await loadStats();
  await loadList();

  // Step 1: kick off session-badge + health-banner pollers. They
  // self-reschedule on success and on error, so they never go silent.
  pollSessionBadge();
  pollHealth();
  pollHealthNow();
})();

// ---- Step 1 — A4 (session clock badge) + A5 (health banner) ----

function formatHours(h) {
  if (!h || h <= 0) return '0m';
  if (h < 1/60) return '<1m';
  const totalMin = Math.round(h * 60);
  const hh = Math.floor(totalMin / 60);
  const mm = totalMin % 60;
  if (hh === 0) return `${mm}m`;
  if (mm === 0) return `${hh}h`;
  return `${hh}h ${mm}m`;
}

function formatAgo(iso) {
  if (!iso) return 'never';
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return 'never';
  const s = Math.max(0, Math.round((Date.now() - t) / 1000));
  if (s < 5) return 'just now';
  if (s < 60) return `${s}s ago`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  const mm = m % 60;
  if (h < 24) return mm ? `${h}h ${mm}m ago` : `${h}h ago`;
  const d = Math.floor(h / 24);
  return `${d}d ago`;
}

let sessionPollTimer = null;
let healthPollTimer = null;
async function pollSessionBadge() {
  const el = document.getElementById('session-badge');
  const label = el.querySelector('.label');
  try {
    const cfg = await api('/api/decay/config');
    const ss = cfg.session_state || {};
    const acc = ss.accumulated_session_hours || 0;
    const alive = !!ss.session_alive;
    const last = ss.session_last_tick_at;
    el.classList.remove('ok', 'warn', 'down', 'idle');

    if (!ss.session_started_at) {
      el.classList.add('idle');
      label.textContent = 'no session yet';
      el.title = 'Session clock hasn’t started. Trigger any tool call or refresh to begin.';
    } else if (alive) {
      el.classList.add('ok');
      label.textContent = `session · ${formatHours(acc)} logged`;
      el.title = `Started ${ss.session_started_at}\nLast tick ${formatAgo(last)}\nThis is the canon for all decay + pop math.`;
    } else {
      el.classList.add('warn');
      label.textContent = `idle · last tick ${formatAgo(last)}`;
      el.title = `No activity for > session_idle_minutes. Will restart on next tool call.\nTotal logged: ${formatHours(acc)}.`;
    }
  } catch (e) {
    el.classList.remove('ok', 'warn', 'down');
    el.classList.add('down');
    label.textContent = 'session · ?';
    el.title = 'Could not reach /api/decay/config: ' + e;
  }
  clearTimeout(sessionPollTimer);
  sessionPollTimer = setTimeout(pollSessionBadge, 10000);
}

function setHBPill(state, text) {
  const p = document.getElementById('hb-dot');
  p.classList.remove('ok', 'warn', 'down', 'idle');
  p.classList.add(state);
  p.innerHTML = `<span class="dot"></span>${text}`;
}

async function pollHealth() {
  const banner = document.getElementById('health-banner');
  banner.style.display = 'flex';
  let mode = 'down';
  try {
    const [h, s] = await Promise.all([
      api('/api/health'),
      api('/api/stats'),
    ]);
    const segs = s.total_segments || s.count || 0;
    // Mandatory count is real store data — show it in read-only mode too.
    // (An earlier version zeroed it when read-only, which made the banner
    // read "0 mandatory" even though the store had 31.)
    const mandatory = (s.mandatory_count || 0);
    document.getElementById('hb-segs').textContent = segs;
    document.getElementById('hb-mand').textContent = mandatory;
    const wlock = h.writer_lock || {};
    const held = wlock.held_by_us ? 'lock: ours' : (wlock.holder_pid ? `lock: pid ${wlock.holder_pid}` : 'lock: free');
    document.getElementById('hb-lock').textContent = held;
    document.getElementById('hb-lock').classList.remove('muted');
    document.getElementById('hb-embed').textContent = h.embedding_model ? `embed: ${h.embedding_model}` : 'embed: loaded';
    document.getElementById('hb-embed').classList.remove('muted');
    document.getElementById('hb-pid').textContent = `pid: ${h.pid}`;
    document.getElementById('hb-pid').classList.remove('muted');

    // Health state: read-only isn't a problem, it's just a mode.
    if (h.read_only) {
      mode = 'warn';
      setHBPill('warn', 'read-only · browse only');
    } else {
      mode = 'ok';
      setHBPill('ok', 'healthy');
    }
    // Update the section-5 mode status line.
    const ms = document.getElementById('mode-status');
    if (ms) {
      ms.textContent = h.read_only
        ? `read-only · lock held by pid ${h.writer_lock?.holder_pid ?? '?'}`
        : h.backend === 'sqlite' ? 'write mode · shared SQLite owner; handoff disabled' : 'write mode · dashboard owns the lock';
      ms.style.color = h.read_only ? '#d9a441' : '#5fb878';
    }
    // Step 6 (D4): surface any degraded_reasons returned by /api/health.
    // Degraded is orthogonal to read-only: a dashboard can be in write
    // mode and still degraded (e.g. embedding model failed to load).
    applyDegradedState(h.degraded_reasons || []);
  } catch (e) {
    setHBPill('down', 'unreachable');
    mode = 'down';
  }
  banner.classList.remove('ok', 'warn', 'down');
  banner.classList.add(mode);
  clearTimeout(healthPollTimer);
  healthPollTimer = setTimeout(pollHealth, 15000);
}

async function pollHealthNow() {
  // Manual refresh button: poll both right now, regardless of cadence.
  await Promise.all([pollSessionBadge(), pollHealth()]);
  // Reset the auto-poll timers by clearing their pending timeouts is
  // overkill — the next scheduled tick will fire normally.
}

// ---- Step 6 (D4) — degraded-state surface in the health banner ----
// If /api/health reports degraded_reasons, show them in a tooltip on
// the pill and colour it amber. Distinct from read-only (which is a
// normal mode, not a degradation).
function applyDegradedState(reasons) {
  const p = document.getElementById('hb-dot');
  const banner = document.getElementById('health-banner');
  // Remove any previous degraded class to avoid stacking.
  p.classList.remove('degraded');
  if (reasons && reasons.length) {
    p.classList.add('degraded');
    p.title = 'Degraded:\n  • ' + reasons.join('\n  • ');
    banner.title = p.title;
  } else {
    p.title = '';
    banner.title = '';
  }
}
</script>
</body>
</html>
"""


def main():
    """Compatibility entrypoint for the explicitly configured SQLite service."""
    from .sqlite_service import cli
    return cli()


if __name__ == "__main__":
    main()
