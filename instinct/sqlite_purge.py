# SPDX-License-Identifier: MPL-2.0
"""Explicit single-record owner purge, separate from agent deletion and replay."""

from pathlib import Path

from . import fallback_store
from .private_storage import ensure_private_directory, private_file_lock
from .sqlite_records import require_text


def purge_recycled(store, record_id, confirmation):
    if store._read_only:
        raise PermissionError('SQLite observer cannot purge records')
    require_text(record_id, 'record ID', 256)
    if confirmation != record_id:
        raise ValueError('confirm_id must exactly match the record ID')
    pending = Path(store.persist_dir) / 'fallback_pending'
    ensure_private_directory(pending)
    with private_file_lock(pending / '.capture.lock', timeout=0):
        stats = {'skipped': 0}
        records = fallback_store.list_pending(str(store.persist_dir), _stats=stats)
        if records or stats['skipped']:
            raise RuntimeError('Pending files require replay or owner review before purge')
        return store.repository.purge_record(record_id)
