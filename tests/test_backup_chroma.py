# SPDX-License-Identifier: MPL-2.0
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from instinct import backup


PROBE = r'''
import json
from pathlib import Path
import sys

from instinct.embedding_provider import EmbeddingProvider
from instinct.models import ScoredSegment, Segment, SegmentType
from instinct.private_storage import write_private_text
from instinct.store import InstinctStore


class SyntheticEmbedding(EmbeddingProvider):
    @staticmethod
    def name():
        return 'synthetic-backup-test'

    def __call__(self, input):
        return [[1.0, float(text.rsplit(' ', 1)[1].rstrip('.')) + 1.0, 0.5] for text in input]


InstinctStore._get_embedding_function = lambda self: SyntheticEmbedding()
root = Path(sys.argv[1])
memory_count = int(sys.argv[3])
if sys.argv[4] == 'installed':
    import instinct
    assert Path(instinct.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
store = InstinctStore(persist_dir=str(root))
if sys.argv[2] == 'create':
    notes = []
    for index in range(memory_count):
        score = (70, 85)[index % 2]
        note = ScoredSegment(
            segment=Segment(content=f'Synthetic backup memory {index}.', segment_type=SegmentType.CONTEXT),
            importance_score=score,
            concept_title=f'Synthetic backup title {index}',
            memory_identifiers=[f'synthetic backup {index}'],
        )
        notes.append(note)
        if len(notes) == 100:
            store.add_batch(notes)
            notes = []
    store.add_batch(notes)
    write_private_text(root / 'synthetic-state.json', '{"synthetic":true}')
records = store._collection.get(include=['documents', 'metadatas', 'embeddings'])
snapshot = []
for index, segment_id in enumerate(records['ids']):
    note = store.get(segment_id)
    assert note is not None
    assert note.importance_score in (70, 85)
    snapshot.append({
        'id': segment_id,
        'document': records['documents'][index],
        'metadata': records['metadatas'][index],
        'embedding': list(map(float, records['embeddings'][index])),
        'title': note.concept_title,
        'score': note.importance_score,
        'initial_score': note.initial_score,
        'identifiers': note.memory_identifiers,
    })
assert store.count() == memory_count
assert len(store.search_by_identifiers(['synthetic backup 0'])) == 1
query_count = min(10, memory_count)
query = store._collection.query(query_embeddings=[[1.0, 2.0, 0.5]], n_results=query_count)
assert len(query['ids'][0]) == query_count
assert set(query['ids'][0]).issubset(records['ids'])
distances = dict(zip(query['ids'][0], query['distances'][0]))
print(json.dumps({
    'records': sorted(snapshot, key=lambda record: record['id']),
    'query_distances': distances,
}, sort_keys=True))
'''


@pytest.mark.skipif(sys.platform != 'linux', reason='Offline recovery supports Linux/WSL only')
@pytest.mark.parametrize('memory_count,installed', [(2, False), (1000, False), (1000, True)])
def test_real_chroma_offline_restore_reopens_in_fresh_process(tmp_path, memory_count, installed):
    executable = os.environ.get('INSTINCT_RECOVERY_PYTHON') if installed else sys.executable
    if installed and not executable:
        pytest.skip('Set INSTINCT_RECOVERY_PYTHON to an installed Linux/WSL runtime')
    source = tmp_path / 'source'
    source.mkdir(mode=0o700)
    snapshot = tmp_path / 'snapshot'
    restored = tmp_path / 'restored'
    environment = {**os.environ, 'HF_HUB_OFFLINE': '1', 'ANONYMIZED_TELEMETRY': 'False'}
    command = [executable, '-I'] if installed else [executable]

    def run(arguments):
        result = subprocess.run(
            [*command, *arguments], cwd=Path(__file__).resolve().parents[1],
            env=environment, capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout.strip().splitlines()[-1])

    def probe(directory, operation):
        return run(['-c', PROBE, str(directory), operation, str(memory_count),
                    'installed' if installed else 'source'])

    def recover(operation, directory, destination=None):
        if installed:
            arguments = ['-m', 'instinct.backup', operation, '--source', str(directory)]
            if destination is not None:
                arguments.extend(['--destination', str(destination), '--confirm-offline'])
            return run(arguments)
        if operation == 'backup':
            return backup.create_backup(directory, destination)
        if operation == 'restore':
            return backup.restore_backup(directory, destination)
        return backup.verify(directory)

    expected = probe(source, 'create')
    assert recover('backup', source, snapshot)['ok']
    if installed:
        assert recover('verify', snapshot)['ok']
    manifest = backup.verify(snapshot)
    assert 'chroma.sqlite3' in manifest['files']
    assert 'synthetic-state.json' in manifest['files']
    assert recover('restore', snapshot, restored)['ok']
    assert probe(restored, 'read') == expected
    assert (restored / 'synthetic-state.json').read_text() == '{"synthetic":true}'
    assert backup.verify(snapshot) == manifest
