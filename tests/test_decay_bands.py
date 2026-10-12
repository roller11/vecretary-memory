# SPDX-License-Identifier: MPL-2.0
from datetime import datetime
from pathlib import Path

import pytest

from instinct import dashboard, dormancy, pop
from instinct.models import ScoredSegment, Segment, SegmentType


@pytest.mark.parametrize('score, expected', [
    (0, 'tail'), (49, 'tail'), (50, 'mid'), (79, 'mid'),
    (80, 'plateau'), (84, 'plateau'), (89, 'plateau'),
    (90, 'plateau'), (91, 'plateau'), (98, 'plateau'),
    (99, 'mandatory'), (100, 'mandatory'),
])
def test_decay_and_pop_share_band_boundaries(score, expected):
    assert dormancy._band_for(score) == expected
    assert pop._band_for_score(score) == expected


def synthetic_note(score):
    return ScoredSegment(
        segment=Segment(content='Synthetic band regression', segment_type=SegmentType.CONTEXT),
        importance_score=score, concept_title='Synthetic band regression',
    )


@pytest.mark.parametrize('score', [80, 84, 89])
def test_plateau_default_reaches_sixty_without_changing_initial_drop(score):
    cfg = dormancy.DormancyConfig()
    memory = synthetic_note(score)
    assert dormancy._band_decay(memory, score, datetime.now(), cfg, projected_hours=0) == score
    assert dormancy._band_decay(memory, score, datetime.now(), cfg, projected_hours=14) == score - 10
    assert dormancy._band_decay(memory, score, datetime.now(), cfg, projected_hours=254) == 60


@pytest.mark.parametrize('score, floor, hours', [(90, 60, 200), (94, 29, 500), (95, 60, 1000)])
def test_special_tiers_keep_their_existing_curves(score, floor, hours):
    cfg = dormancy.DormancyConfig()
    memory = synthetic_note(score)
    assert dormancy._band_decay(memory, score, datetime.now(), cfg, projected_hours=hours) == floor


@pytest.mark.parametrize('band', ['plateau', 'mid'])
def test_dashboard_accepts_both_decay_periods_and_floor(band):
    cfg = dormancy.DormancyConfig()
    updates = dashboard._validated_config_updates({band: {
        'initial_drop_active_hours': 12.0,
        'post_drop_to_floor_active_hours': 300.0,
        'post_drop_half_life_active_hours': 40.0,
        'floor': 60 if band == 'plateau' else 20,
    }}, cfg)
    assert updates[band]['initial_drop_active_hours'] == 12.0
    assert updates[band]['post_drop_to_floor_active_hours'] == 300.0
    assert updates[band]['post_drop_half_life_active_hours'] == 40.0


def test_dashboard_explains_new_ranges():
    source = Path(dashboard.__file__).read_text(encoding='utf-8')
    assert 'plateau (80-89; upper standard fallback)' in source
    assert 'mid (50-79)' in source
