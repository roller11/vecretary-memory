# SPDX-License-Identifier: MPL-2.0
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import threading
from unittest.mock import Mock
from types import SimpleNamespace

import pytest

from instinct import dashboard, dormancy
from instinct.models import ScoredSegment, Segment, SegmentType
from instinct.private_storage import ensure_private_directory
from tests.test_dashboard_security import request
from tests.test_identifier_recycling import SyntheticCollection, make_store


class FeatureCollection(SyntheticCollection):
    def get(self, ids=None, limit=None, offset=0, include=None, where=None):
        if where is not None:
            minimum = where['$and'][0]['importance_score']['$gte']
            maximum = where['$and'][1]['importance_score']['$lte']
            ids = [segment_id for segment_id, record in self.records.items()
                   if minimum <= record['metadata']['importance_score'] <= maximum]
        return super().get(ids=ids, limit=limit, offset=offset, include=include)

    def delete(self, ids):
        for segment_id in ids:
            self.records.pop(segment_id, None)

    def update(self, ids, metadatas):
        for segment_id, metadata in zip(ids, metadatas):
            self.records[segment_id]['metadata'].update(metadata)


@pytest.fixture
def feature_dashboard(tmp_path, monkeypatch):
    root = ensure_private_directory(tmp_path / 'synthetic-dashboard')
    store = make_store(FeatureCollection(root))
    store.persist_dir = str(root)
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


def payload(**updates):
    return {'title': 'Synthetic feature fixture', 'content': 'Synthetic orchard test memory.',
            'score': 70, 'segment_type': 'context', 'decay_profile': 'standard', **updates}


def test_observer_cannot_initialize_writer_during_dashboard_release(monkeypatch):
    entered = threading.Event()
    finish = threading.Event()
    observing = threading.Event()
    observed = threading.Event()
    modes = []

    def construct(**kwargs):
        modes.append(kwargs['read_only'])
        if kwargs['read_only']:
            entered.set()
            assert finish.wait(timeout=5)
        return SimpleNamespace(_read_only=kwargs['read_only'], _writer_lock_handle=None)

    def observe():
        observing.set()
        store = dashboard.get_store()
        observed.set()
        return store

    monkeypatch.setattr(dashboard, '_store', SimpleNamespace(_writer_lock_handle=None))
    monkeypatch.setattr(dashboard, '_retriever', None)
    monkeypatch.setattr(dashboard, 'InstinctStore', construct)
    monkeypatch.setattr(dashboard, 'PERSIST_DIR', 'synthetic-transition-fixture')
    monkeypatch.setattr('instinct.configure_dormancy', lambda directory: None)
    with ThreadPoolExecutor(max_workers=2) as pool:
        release = pool.submit(dashboard.release_write_mode)
        try:
            assert entered.wait(timeout=5)
            observer = pool.submit(observe)
            assert observing.wait(timeout=5)
            assert not observed.wait(timeout=0.05)
        finally:
            finish.set()
        assert release.result(timeout=5)['released']
        assert observer.result(timeout=5)._read_only
    assert modes == [True]


@pytest.mark.parametrize('include_recycle', [False, True])
def test_reset_real_store_contract_and_recycle_snapshot(feature_dashboard, include_recycle):
    server, store, root = feature_dashboard
    active = ScoredSegment(Segment('Synthetic active reset fixture', SegmentType.CONTEXT), 70, 'Active reset')
    recycled = ScoredSegment(Segment('Synthetic recycled reset fixture', SegmentType.CONTEXT), 60, 'Recycled reset')
    store.add(active)
    store.add(recycled)
    store.recycle(recycled.id)
    status, preview = post(server, '/api/admin/reset', {'dry_run': True, 'include_recycle': include_recycle})
    assert status == 200 and preview['active_segment_count'] == 1
    assert store.count() == 2 and not (root / 'backups').exists()
    status, result = post(server, '/api/admin/reset', {'include_recycle': include_recycle})
    assert status == 200 and result['ok']
    assert result['wiped_active'] == 1 and result['wiped_recycle'] == int(include_recycle)
    assert store.get(active.id) is None
    assert (store.get(recycled.id) is None) == include_recycle
    snapshot = json.loads(Path(result['backup_recycle']).read_text())
    assert snapshot[0]['id'] == recycled.id
    assert snapshot[0]['metadata']['content'] == recycled.segment.content


def test_reset_reports_failed_deletion_instead_of_success(feature_dashboard, monkeypatch):
    server, store, _ = feature_dashboard
    active = ScoredSegment(Segment('Synthetic failed reset fixture', SegmentType.CONTEXT), 70, 'Failed reset')
    store.add(active)
    monkeypatch.setattr(store, 'delete', lambda segment_id: False)
    status, result = post(server, '/api/admin/reset', {})
    assert status == 500 and 'not confirmed' in result['error']
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
