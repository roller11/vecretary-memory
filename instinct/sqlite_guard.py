# SPDX-License-Identifier: MPL-2.0
"""SQLite-specific schema validation and lifetime advisory writer ownership."""

import json
import math

from .private_storage import check_private_file, private_binary_reader, private_file_lock
from .sqlite_records import EmbeddingIdentity


APPLICATION_ID = 0x564D5351
SCHEMA_VERSION = 1
WRITER_LOCK_FILENAME = '.sqlite.writer.lock'
SCHEMA = {
    'identity': 'CREATE TABLE identity (singleton INTEGER PRIMARY KEY CHECK(singleton = 1), value TEXT NOT NULL)',
    'records': 'CREATE TABLE records (id TEXT PRIMARY KEY NOT NULL, content TEXT NOT NULL, metadata TEXT NOT NULL, embedding BLOB NOT NULL)',
}


class SQLiteWriterUnavailable(RuntimeError):
    pass


def acquire_writer_guard(root, timeout=0):
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 <= timeout <= 5:
        raise ValueError('Writer timeout must be a finite value from 0 to 5 seconds')
    guard = private_file_lock(root / WRITER_LOCK_FILENAME, timeout=timeout)
    try:
        guard.__enter__()
    except TimeoutError:
        raise SQLiteWriterUnavailable('SQLite writer is owned by another session') from None
    return guard


def validate_writable_header(path):
    with private_binary_reader(path) as source:
        header = source.read(100)
    if len(header) != 100 or header[:16] != b'SQLite format 3\x00':
        raise ValueError('Foreign SQLite database')
    if int.from_bytes(header[68:72], 'big') != APPLICATION_ID:
        raise ValueError('Foreign SQLite database')
    if int.from_bytes(header[60:64], 'big') != SCHEMA_VERSION:
        raise ValueError('Unsupported SQLite schema version')
    if header[18:20] != b'\x01\x01':
        raise ValueError('Only rollback-journal SQLite databases are supported')
    for suffix in ('-wal', '-shm'):
        sidecar = path.with_name(path.name + suffix)
        if sidecar.exists() or sidecar.is_symlink():
            raise ValueError('Unsupported SQLite WAL sidecar')
    journal = path.with_name(path.name + '-journal')
    if journal.exists() or journal.is_symlink():
        check_private_file(journal)


def validate_database(connection, identity):
    if connection.execute('PRAGMA application_id').fetchone() != (APPLICATION_ID,):
        raise ValueError('Foreign SQLite database')
    if connection.execute('PRAGMA user_version').fetchone() != (SCHEMA_VERSION,):
        raise ValueError('Unsupported SQLite schema version')
    schema = connection.execute("SELECT name, type, sql FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'").fetchall()
    if sorted(schema) != sorted((name, 'table', statement) for name, statement in SCHEMA.items()):
        raise ValueError('Unsupported SQLite record schema')
    rows = connection.execute('SELECT value FROM identity').fetchall()
    if len(rows) != 1 or EmbeddingIdentity.from_dict(json.loads(rows[0][0])) != identity:
        raise ValueError('Embedding identity mismatch')
    if connection.execute('PRAGMA quick_check').fetchone() != ('ok',):
        raise ValueError('SQLite integrity check failed')
