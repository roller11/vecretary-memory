# SPDX-License-Identifier: MPL-2.0
"""Controlled-clock decay and real process-restart checks against an installed wheel."""

from datetime import datetime, timedelta
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def close_to(actual, expected, message):
    require(math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-8), message)


def restart_probe(data):
    from instinct import session_timer

    before = session_timer.state(data).accumulated_session_seconds
    tick = session_timer.tick_session(data)
    require(tick.seconds_added_to_accumulator == 0, 'Restart charged an offline gap')
    close_to(tick.state.accumulated_session_seconds, before, 'Restart changed accumulated active time')
    print(json.dumps({'first_tick_added_seconds': 0, 'accumulated_seconds': before}))


def run(root):
    from instinct import session_timer
    from instinct.active_time import capture_anchors, elapsed_hours
    from instinct.decay import DecayModifier
    from instinct.dormancy import config_for, current_effective_score
    from instinct.models import ScoredSegment, Segment, SegmentType
    from instinct.modifiers import QueryContext, ScoringResult
    from instinct.pop import apply_pop

    data = root / 'controlled-clock'
    data.mkdir(mode=0o700)
    start = datetime(2026, 10, 3, 10)
    session_timer.reset_session(data)
    session_timer.tick_session(data, now=start)
    config = config_for(data)
    memories = [ScoredSegment(
        segment=Segment(content='Synthetic installed decay note', segment_type=SegmentType.CONTEXT),
        importance_score=score, concept_title=f'Synthetic score {score}',
    ) for score in (40, 70, 90)]
    for memory in memories:
        capture_anchors(memory, data, new=True)
    for minute in range(1, 301):
        session_timer.tick_session(data, now=start + timedelta(minutes=minute))
    end = start + timedelta(hours=5)
    scores = []
    for memory in memories:
        close_to(elapsed_hours(memory, 'created', data), 5, 'Wrong per-memory active age')
        original = memory.importance_score
        expected = current_effective_score(memory, end, config)
        require(expected < original, 'Configured decay curve remained flat')
        result = DecayModifier().modify(memory, QueryContext(now=end, persist_dir=str(data)),
                                        ScoringResult(memory.id, original, float(original)))
        close_to(result.effective_score, expected, 'Retrieval and dashboard scoring differ')
        memory.importance_score = round(expected)
        close_to(current_effective_score(memory, end, config), expected, 'Repeated decay reapplied total age')
        restored = ScoredSegment.from_storage_dict(memory.to_storage_dict())
        close_to(elapsed_hours(restored, 'created', data), 5, 'Storage roundtrip lost active anchor')
        scores.append({'initial_score': original, 'effective_score': expected})
    gap = end + timedelta(days=7)
    tick = session_timer.tick_session(data, now=gap)
    require(tick.seconds_added_to_accumulator == 0, 'Idle week charged to active time')
    for memory, score in zip(memories, scores):
        close_to(current_effective_score(memory, gap, config), score['effective_score'], 'Idle week changed decay')
    new_memory = ScoredSegment(
        segment=Segment(content='Synthetic new anchor', segment_type=SegmentType.CONTEXT),
        importance_score=70, concept_title='Synthetic new anchor',
    )
    capture_anchors(new_memory, data, new=True)
    close_to(elapsed_hours(new_memory, 'created', data), 0, 'New note inherited earlier activity')
    popped = apply_pop(memories[1], 2, config, now=gap)
    require(popped.rule_applied == 'halve', 'Expected active-age halving pop')
    from instinct.dormancy import _active_age
    close_to(_active_age(memories[1], gap, config), 2.5, 'Pop did not halve active age')

    real_data = root / 'real-clock'
    real_data.mkdir(mode=0o700)
    session_timer.tick_session(real_data)
    time.sleep(1.1)
    tick = session_timer.tick_session(real_data)
    require(tick.seconds_added_to_accumulator > 0, 'Real monotonic activity did not advance')
    child = subprocess.run([sys.executable, '-I', str(Path(__file__).resolve()), '--restart-child', str(real_data)],
                           cwd=root, capture_output=True, text=True, check=True, timeout=60)
    restart = json.loads(child.stdout)
    wall_data = root / 'wall-clock-jump'
    wall_data.mkdir(mode=0o700)
    with patch.object(session_timer, 'datetime') as wall, patch.object(
        session_timer.time, 'monotonic', side_effect=[100.0, 160.0, 220.0]
    ):
        wall.now.side_effect = [start, start + timedelta(days=3650), start - timedelta(days=3650)]
        wall.fromisoformat = datetime.fromisoformat
        for index in range(3):
            session_timer.tick_session(wall_data)
    close_to(session_timer.state(wall_data).accumulated_session_seconds, 120, 'Wall clock affected accumulator')
    return {
        'status': 'passed', 'installed_module': __import__('instinct').__file__,
        'controlled_clock': {'active_hours': 5, 'scores': scores, 'idle_week_excluded': True,
                             'per_memory_anchors': 'passed', 'repeated_decay': 'passed',
                             'retrieval_dashboard_parity': 'passed', 'pop_halves_active_age': True,
                             'wall_clock_jumps_excluded': True},
        'real_monotonic_activity': 'passed', 'real_process_restart': restart,
        'real_ten_minute_idle': 'not_tested_here_use_soak_stage',
        'editor_model_behavior': 'not_certified',
    }


def main():
    import instinct

    require(sys.platform == 'linux' and 'site-packages' in Path(instinct.__file__).resolve().parts,
            'Use a wheel-installed Linux/WSL interpreter outside the checkout')
    if len(sys.argv) > 1:
        require(len(sys.argv) == 3 and sys.argv[1] == '--restart-child', 'Invalid child arguments')
        restart_probe(Path(sys.argv[2]))
        return
    root = Path(tempfile.mkdtemp(prefix='instinct-decay-acceptance-'))
    report = {'status': 'running', 'synthetic_root': str(root)}
    try:
        report.update(run(root))
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        (root / 'decay.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
