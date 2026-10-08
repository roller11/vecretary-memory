# SPDX-License-Identifier: MPL-2.0
"""Verify a cold full-directory roundtrip on disposable installed synthetic data."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def environment_for(data):
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(('CE_', 'INSTINCT_', 'PYTHON'))}
    environment.update({
        'INSTINCT_DATA_DIR': str(data), 'CE_DATA_DIR': str(data),
        'CE_EMBEDDING_PROVIDERS': 'onnx', 'INSTINCT_ALLOW_REMOTE_PROVIDERS': '0',
        'INSTINCT_ALLOW_INSECURE_REMOTE_HTTP': '0', 'PYTHONDONTWRITEBYTECODE': '1',
    })
    return environment


def probe(mode, data):
    from instinct.active_time import elapsed_hours
    from instinct.models import ScoredSegment, Segment, SegmentType
    from instinct.store import InstinctStore

    store = InstinctStore(persist_dir=str(data))
    if mode == 'create':
        from instinct import session_timer
        from instinct.private_storage import write_private_text

        session_timer.tick_session(data)
        for title in ('Synthetic retained note', 'Synthetic recycled note'):
            memory = ScoredSegment(Segment('Synthetic recovery content: ' + title, SegmentType.CONTEXT),
                                   importance_score=70, concept_title=title)
            store.add(memory)
            if 'recycled' in title:
                require(store.recycle(memory.id, by='synthetic-owner', reason='Recovery fixture') is not None,
                        'Fixture recycling failed')
        write_private_text(data / 'synthetic-recovery-state.json', '{"synthetic": true}')
        from datetime import datetime, timedelta

        session_timer.tick_session(data, now=datetime.now() + timedelta(minutes=5))
    records = store._collection.get(include=['documents', 'metadatas', 'embeddings'])
    rows = []
    for index, identifier in enumerate(records['ids']):
        memory = store.get(identifier)
        rows.append({
            'id': identifier, 'document': records['documents'][index],
            'metadata': records['metadatas'][index],
            'embedding': records['embeddings'][index].tolist(),
            'created_active_hours': elapsed_hours(memory, 'created', data),
        })
    matches = store.search('Synthetic retained recovery content', limit=2, min_importance=0)
    require(len(rows) == 2 and len(matches) == 1, 'Active/recycled fixture or search exclusion mismatch')
    print(json.dumps({'rows': sorted(rows, key=lambda row: row['id']),
                      'search_ids': [memory.id for memory in matches]}, sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm-synthetic-restore', action='store_true')
    parser.add_argument('--product-backup', action='store_true', help='Exercise the installed owner backup/restore implementation')
    arguments = parser.parse_args(argv)
    if not arguments.confirm_synthetic_restore:
        parser.error('Authorize fresh synthetic data only with --confirm-synthetic-restore')
    import instinct

    if sys.platform != 'linux' or 'site-packages' not in Path(instinct.__file__).resolve().parts:
        parser.error('Use a wheel-installed Linux/WSL interpreter outside the checkout')
    root = Path(tempfile.mkdtemp(prefix='instinct-restore-acceptance-'))
    source, backup, restored = (root / name for name in ('source', 'cold-backup', 'restored'))
    report = {'status': 'running', 'synthetic_root': str(root),
              'installed_module': instinct.__file__, 'live_stores_accessed': False,
              'method': 'product_offline_backup_restore' if arguments.product_backup else 'cold_full_directory_copy_after_owned_process_exit',
              'dashboard_reset_snapshot_restore': 'not_certified',
              'online_backup_and_cross_version_restore': 'not_tested'}

    def child(mode, data):
        completed = subprocess.run(
            [sys.executable, '-I', '-c',
             "import runpy,sys; sys.path.insert(0,sys.argv[1]); sys.argv=sys.argv[2:]; runpy.run_path(sys.argv[0],run_name='__main__')",
             str(Path(instinct.__file__).resolve().parent.parent), str(Path(__file__).resolve()), '--probe', mode],
            cwd=root, env=environment_for(data), capture_output=True, text=True, timeout=90,
        )
        require(completed.returncode == 0, 'Synthetic probe failed: ' + completed.stderr)
        return json.loads(completed.stdout)

    try:
        before = child('create', source)
        from instinct.chroma_writer_lock import is_writer_lock_held

        require(not is_writer_lock_held(str(source)), 'Owned writer did not release; do not copy a live store')
        if arguments.product_backup:
            from instinct.backup import create_backup, restore_backup, verify

            create_backup(source, backup)
            verify(backup)
        else:
            shutil.copytree(source, backup)
        backup_files = {path.relative_to(backup): path.read_bytes() for path in backup.rglob('*') if path.is_file()}
        if arguments.product_backup:
            restore_backup(backup, restored)
        else:
            shutil.copytree(backup, restored)
        after = child('inspect', restored)
        require(before == after, 'Restored documents, metadata, vectors, active anchors or search results differ')
        require((restored / 'synthetic-recovery-state.json').read_text() == '{"synthetic": true}',
                'Auxiliary state was not restored')
        require(all((backup / path).read_bytes() == content for path, content in backup_files.items()),
                'Restore changed the backup')
        require(not is_writer_lock_held(str(restored)), 'Restored owned writer did not release')
        report.update(status='passed', records=len(after['rows']), vector_and_metadata_parity=True,
                      semantic_search_parity=True, active_anchor_parity=True,
                      auxiliary_state_restored=True, backup_unchanged=True, owned_writers_released=True)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        from instinct.private_storage import write_private_text

        write_private_text(root / 'restore.json', json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--probe' and sys.argv[2] in ('create', 'inspect'):
        probe(sys.argv[2], Path(os.environ['INSTINCT_DATA_DIR']))
    else:
        main()
