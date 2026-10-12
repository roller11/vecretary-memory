# SPDX-License-Identifier: MPL-2.0
from datetime import datetime
from copy import deepcopy
from pathlib import Path
import json
import threading
from unittest.mock import Mock
from types import SimpleNamespace

import pytest

from instinct import dashboard, dormancy
from instinct.models import RECYCLED_AT_KEY, ScoredSegment, Segment, SegmentType
from instinct.private_storage import ensure_private_directory
from tests.test_dashboard_security import request
class FeatureStore:
    def __init__(self, persist_dir):
        self.persist_dir = str(persist_dir)
        self._read_only = False
        self.memories = {}

    def add(self, segment):
        if self._read_only:
            raise RuntimeError('Store is read-only')
        self.memories[segment.id] = deepcopy(segment)
        return segment.id

    def add_batch(self, segments):
        return [self.add(segment) for segment in segments]

    def get(self, segment_id):
        segment = self.memories.get(segment_id)
        return deepcopy(segment) if segment is not None else None

    def get_many(self, segment_ids):
        return [self.get(segment_id) for segment_id in segment_ids]

    def get_by_score(self, min_score=0, max_score=100, limit=100, include_recycled=False):
        selected = [segment for segment in self.memories.values()
                    if min_score <= segment.importance_score <= max_score
                    and (include_recycled or not segment.segment.metadata.get(RECYCLED_AT_KEY))]
        return [deepcopy(segment) for segment in sorted(
            selected, key=lambda item: item.importance_score, reverse=True)[:limit]]

    def get_mandatory(self, include_recycled=False):
        return self.get_by_score(99, 100, limit=len(self.memories), include_recycled=include_recycled)

    def list_recycled(self, limit=200):
        return [deepcopy(segment) for segment in self.memories.values()
                if segment.segment.metadata.get(RECYCLED_AT_KEY)][:limit]

    def count(self):
        return len(self.memories)

    def recycle(self, segment_id, by='dashboard', reason=''):
        segment = self.memories.get(segment_id)
        if segment is None or self._read_only:
            return None
        segment = deepcopy(segment)
        segment.segment.metadata[RECYCLED_AT_KEY] = datetime.now().isoformat()
        self.memories[segment_id] = segment
        return deepcopy(segment)

    def restore(self, segment_id):
        segment = self.memories.get(segment_id)
        if segment is None or self._read_only:
            return None
        segment = deepcopy(segment)
        segment.segment.metadata.pop(RECYCLED_AT_KEY, None)
        self.memories[segment_id] = segment
        return deepcopy(segment)

    def delete(self, segment_id):
        if self._read_only:
            return False
        return self.memories.pop(segment_id, None) is not None

    def update_metadata_only(self, segment):
        if self._read_only or segment.id not in self.memories:
            return False
        self.memories[segment.id] = deepcopy(segment)
        return True

    def update_score(self, segment_id, score, reasoning=''):
        segment = self.get(segment_id)
        if segment is None or self._read_only:
            return False
        segment.importance_score = score
        segment.reasoning = reasoning
        return self.update_metadata_only(segment)

    def search(self, query, limit=10, min_importance=None, include_recycled=False, **_):
        terms = set(query.lower().split())
        found = [segment for segment in self.get_by_score(
            min_score=0 if min_importance is None else min_importance,
            limit=len(self.memories), include_recycled=include_recycled)
            if terms & set((segment.concept_title + ' ' + segment.segment.content).lower().split())]
        return found[:limit]


@pytest.fixture
def feature_dashboard(tmp_path, monkeypatch):
    root = ensure_private_directory(tmp_path / 'synthetic-dashboard')
    store = FeatureStore(root)
    monkeypatch.setattr(dormancy, '_store', dormancy._ConfigStore(root))
    monkeypatch.setattr(dashboard, 'PERSIST_DIR', str(root))
    monkeypatch.setattr(dashboard, 'get_store', lambda: store)
    monkeypatch.setattr(dashboard, 'is_read_only', lambda: store._read_only)
    retriever = Mock()
    retriever.retrieve.return_value = Mock(mandatory=[], relevant=[], pop_results=[])
    monkeypatch.setattr(dashboard, 'get_retriever', lambda: retriever)
    server = dashboard.DashboardHTTPServer(('127.0.0.1', 0), dashboard.Handler)
    worker = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.01})
    worker.start()
    try:
        yield server, store, root
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)
        assert not worker.is_alive()


def post(server, route, payload):
    status, body, _ = request(server, route=route, body=json.dumps(payload).encode())
    return status, json.loads(body)


def test_dashboard_header_and_logo_use_packaged_same_origin_asset(feature_dashboard):
    server, _, _ = feature_dashboard
    status, body, headers = request(server, method='GET', route='/')
    assert status == 200
    assert b'<img class="brand-mark" src="/assets/vecretary-logo-256.png" alt="Vecretary Memory">' in body
    assert b"prefers-reduced-motion: reduce" in body
    assert b"--surface-2" in body
    assert "img-src 'self' data:" in headers['Content-Security-Policy']

    status, image, headers = request(server, method='GET', route='/assets/vecretary-logo.png')
    assert status == 200
    assert headers['Content-Type'] == 'image/png'
    assert headers['X-Content-Type-Options'] == 'nosniff'
    assert image == dashboard.resources.files('instinct').joinpath(
        'assets', 'vecretary-logo-512.png').read_bytes()

    status, image, headers = request(server, method='GET', route='/assets/vecretary-logo-256.png')
    assert status == 200
    assert headers['Content-Type'] == 'image/png'
    assert headers['X-Content-Type-Options'] == 'nosniff'
    assert image == dashboard.resources.files('instinct').joinpath(
        'assets', 'vecretary-logo-256.png').read_bytes()


@pytest.mark.parametrize('route', ['/', '/score', '/decay'])
def test_dashboard_pages_share_nonsticky_header_and_navigation(feature_dashboard, route):
    server, _, _ = feature_dashboard
    status, body, _ = request(server, method='GET', route=route)
    assert status == 200
    assert b'dashboardInstallSharedHeader' in body
    assert b'position: static !important;' in body
    assert b'height: 280px;' in body
    assert b'/assets/vecretary-logo-256.png' in body
    assert b'--header-gradient-start' in body
    assert b'--header-gradient-mid' in body
    assert b'--header-gradient-end' in body
    assert b'Primary dashboard' in body
    assert b'Score helper' in body


def test_decay_plot_and_settings_share_palette(feature_dashboard):
    server, _, _ = feature_dashboard
    status, body, _ = request(server, method='GET', route='/decay')
    assert status == 200
    for score in (100, 90, 80, 70, 50, 20):
        assert f"color: 'var(--decay-score-{score})'".encode() in body
    assert b'border: 2px solid var(--range-color)' in body
    assert b'fieldParent.dataset.range = ' in body
    assert b'stroke-dasharray=' in body


def payload(**updates):
    return {'title': 'Synthetic feature fixture', 'content': 'Synthetic orchard test memory.',
            'score': 70, 'segment_type': 'context', 'decay_profile': 'standard', **updates}


def test_dashboard_refuses_storage_without_sqlite_service_binding():
    from instinct.service_binding import bind_store

    with pytest.raises(RuntimeError, match='only inside the SQLite service'):
        dashboard.get_store()

    with bind_store(SimpleNamespace(persist_dir='synthetic-runtime', _read_only=True)) as store:
        assert dashboard.get_store() is store


@pytest.mark.parametrize('include_recycle', [False, True])
def test_reset_is_disabled_without_mutating_store(feature_dashboard, include_recycle):
    server, store, root = feature_dashboard
    active = ScoredSegment(Segment('Synthetic reset refusal fixture', SegmentType.CONTEXT), 70, 'Reset refusal')
    store.add(active)
    before = store.count()
    status, result = post(server, '/api/admin/reset', {'include_recycle': include_recycle})
    assert status == 501 and result['code'] == 501
    assert store.count() == before and store.get(active.id) is not None
    assert not (root / 'backups').exists()


def test_reset_refusal_does_not_delete_memory(feature_dashboard):
    server, store, _ = feature_dashboard
    active = ScoredSegment(Segment('Synthetic failed reset fixture', SegmentType.CONTEXT), 70, 'Failed reset')
    store.add(active)
    status, result = post(server, '/api/admin/reset', {})
    assert status == 501 and 'not available' in result['error']
    assert store.get(active.id) is not None


@pytest.mark.parametrize('segment_type', list(SegmentType))
def test_all_public_types_round_trip(feature_dashboard, segment_type):
    server, store, _ = feature_dashboard
    status, result = post(server, '/api/add', payload(segment_type=segment_type.value, score=0))
    assert status == 200
    saved = store.get(result['id'])
    assert saved.segment.segment_type is segment_type
    assert saved.importance_score == (100 if segment_type.value in ('standing_instruction', 'critical_constraint') else 0)


def test_full_edit_preserves_id_history_and_round_trip(feature_dashboard):
    server, store, _ = feature_dashboard
    original = ScoredSegment(Segment('Original synthetic content', SegmentType.CONTEXT), 70, 'Original synthetic title')
    original.access_count = 3
    original.last_accessed = datetime(2026, 1, 1)
    store.add(original)
    status, result = post(server, '/api/update', payload(id=original.id, score=0, title='Edited synthetic title',
        content='Edited synthetic content', project='Synthetic group', tags=['test'], segment_type='user_preference',
        decay_profile='short_term_critical'))
    assert status == 200 and result['id'] == original.id
    saved = store.get(original.id)
    assert saved.id == original.id and store.count() == 1
    assert saved.segment.content == 'Edited synthetic content'
    assert saved.concept_title == 'Edited synthetic title'
    assert saved.importance_score == 0 and saved.initial_score == 70
    assert saved.access_count == 3 and saved.last_accessed == original.last_accessed
    assert saved.segment.segment_type is SegmentType.USER_PREFERENCE
    assert saved.decay_profile == 'short_term_critical'
    assert saved.segment.metadata['project'] == 'Synthetic group'
    assert 'edited' in ' '.join(saved.memory_identifiers)
    assert ScoredSegment.from_storage_dict(saved.to_storage_dict()).id == original.id


def test_profile_pop_and_ephemeral_state_are_durable_and_not_shadowed():
    memory = ScoredSegment(Segment('Synthetic serialization fixture', SegmentType.CONTEXT), 80, 'Synthetic profile')
    memory.decay_profile = 'ephemeral'
    memory.ephemeral_phase = 'fading'
    memory.ephemeral_turns_left = 4
    memory.last_bumped_at = datetime(2026, 1, 1)
    memory.bump_count = 3
    memory.segment.metadata.update({'importance_score': 100, 'decay_profile': 'standard', 'bump_count': 0})
    saved = ScoredSegment.from_storage_dict(memory.to_storage_dict())
    assert saved.importance_score == 80 and saved.decay_profile == 'ephemeral'
    assert saved.ephemeral_phase == 'fading' and saved.ephemeral_turns_left == 4
    assert saved.last_bumped_at == memory.last_bumped_at and saved.bump_count == 3


@pytest.mark.parametrize('updates', [{'score': True}, {'score': '70'}, {'score': -1}, {'score': 101},
    {'score': 2.5}, {'segment_type': 'instruction'}, {'decay_profile': 'invalid'},
    {'tags': [False]}, {'project': []}, {'title': ' '}, {'content': None}])
def test_invalid_add_is_rejected_without_writes(feature_dashboard, updates):
    server, store, _ = feature_dashboard
    assert post(server, '/api/add', payload(**updates))[0] == 400
    assert store.count() == 0


def test_ephemeral_profile_is_not_silently_dropped(feature_dashboard):
    server, store, _ = feature_dashboard
    status, result = post(server, '/api/add', payload(decay_profile='ephemeral'))
    assert status == 200
    saved = store.get(result['id'])
    assert saved.decay_profile == 'ephemeral' and saved.ephemeral_turns_left > 0


def test_complete_config_and_stc_keys_round_trip(feature_dashboard):
    server, _, root = feature_dashboard
    values = dormancy._cfg().to_dict()
    values['mid']['floor'] = 21
    values['stc_hours_by_score'][91] = 4.5
    status, result = post(server, '/api/decay/config', values)
    assert status == 200
    assert result['config']['mid']['floor'] == 21
    reloaded = dormancy._ConfigStore(root).load()
    assert reloaded.mid.floor == 21 and reloaded.stc_hours_by_score[91] == 4.5
    assert reloaded.to_dict() == dormancy._cfg().to_dict()


@pytest.mark.parametrize('updates', [{'obsolete_field': 1}, {'session_idle_minutes': 0},
    {'mid': {'floor': 101}}, {'mid': {'decay_function': 'invalid'}}, {'ephemeral_surface_turns': 1.5},
    {'session_idle_minutes': True}, {'stc_hours_by_score': {'91': 0}}, {'mid': {'unknown': 2}}])
def test_invalid_config_is_atomic(feature_dashboard, updates):
    server, _, root = feature_dashboard
    original = dormancy._cfg().to_dict()
    assert post(server, '/api/decay/config', updates)[0] == 400
    assert dormancy._cfg().to_dict() == original
    assert not (root / dormancy.CONFIG_FILENAME).exists()


@pytest.mark.parametrize('route', ['/api/add', '/api/update', '/api/update_score', '/api/recycle',
    '/api/restore', '/api/purge', '/api/segment/pop', '/api/nudge/run', '/api/nudge/interval',
    '/api/score/ephemeral', '/api/decay/config', '/api/decay/tick', '/api/admin/reset'])
def test_read_only_rejects_every_owner_mutation(feature_dashboard, route):
    server, store, _ = feature_dashboard
    store._read_only = True
    status, result = post(server, route, {})
    assert status == 409 and result['read_only'] is True
    assert store.count() == 0


def test_read_only_search_remains_available_without_mutation(feature_dashboard):
    server, store, _ = feature_dashboard
    store._read_only = True
    assert post(server, '/api/search_with_pop', {'q': 'synthetic', 'apply_pops': False, 'track_access': False})[0] == 200
    assert post(server, '/api/search_with_pop', {'q': 'synthetic', 'apply_pops': True})[0] == 409


def test_mandatory_modifier_and_nudge_controls(feature_dashboard):
    server, store, _ = feature_dashboard
    memory = ScoredSegment(Segment('Synthetic mandatory fixture', SegmentType.STANDING_INSTRUCTION), 100, 'Synthetic mandatory')
    store.add(memory)
    status, body, _ = request(server, method='GET', route='/api/segment/modifier_window?id=' + memory.id, body=None)
    assert status == 200 and json.loads(body)['modifier_window']['sections']
    assert post(server, '/api/nudge/interval', {'id': memory.id, 'turns': 7, 'active': False})[0] == 200
    state = dashboard._seg_to_dict(store.get(memory.id))['nudge']
    assert state['nudge_interval_turns'] == 7 and state['nudge_active'] is False
    assert post(server, '/api/nudge/interval', {'id': memory.id, 'turns': 1, 'active': 'false'})[0] == 400
