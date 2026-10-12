# SPDX-License-Identifier: MPL-2.0
"""Owner-private SQLite record repository for the SQLite runtime."""

import json
import heapq
import math
import os
from pathlib import Path
import sqlite3
import struct

from .private_storage import check_private_file, ensure_private_directory
from .sqlite_records import MAX_RECORD_BYTES, MAX_RECORDS, EmbeddingIdentity, SQLiteRecord, canonical_json, validate_records
from .sqlite_search import VectorHit, validate_vector, vector_distance
from .sqlite_search import prepare_cosine_vector, prepared_cosine_distance
from .sqlite_search_cache import SearchRecordCache
from .sqlite_mutations import SQLiteRecordMutations
from .sqlite_reads import SQLiteRecordReads
from .sqlite_guard import APPLICATION_ID, SCHEMA_VERSION, SCHEMA, acquire_writer_guard, validate_database, validate_writable_header


class SQLiteRecordStore(SQLiteRecordMutations, SQLiteRecordReads):
    def __init__(self, connection, identity, read_only, writer_guard=None):
        self._connection = connection
        self.identity = identity
        self.read_only = read_only
        self._writer_guard = writer_guard
        self._search_cache = SearchRecordCache()

    @staticmethod
    def _connect(path, *, read_only, configure_writer=True):
        check_private_file(path)
        mode = 'ro' if read_only else 'rw'
        connection = sqlite3.connect(path.as_uri() + f'?mode={mode}', uri=True,
                                     isolation_level=None, timeout=5)
        try:
            connection.enable_load_extension(False)
            connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_RECORD_BYTES * 2)
            connection.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 16 * 1024)
            connection.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED, 0)
            connection.execute('PRAGMA trusted_schema = OFF')
            if connection.execute('PRAGMA trusted_schema').fetchone() != (0,):
                raise RuntimeError('SQLite trusted-schema control is unavailable')
            connection.execute('PRAGMA foreign_keys = ON')
            if read_only:
                connection.execute('PRAGMA query_only = ON')
            elif configure_writer:
                SQLiteRecordStore._configure_writer(connection)
            return connection
        except BaseException:
            connection.close()
            raise

    @staticmethod
    def _configure_writer(connection):
        if connection.execute('PRAGMA journal_mode = DELETE').fetchone() != ('delete',):
            raise RuntimeError('Rollback journaling is required')
        connection.execute('PRAGMA synchronous = FULL')
        if connection.execute('PRAGMA synchronous').fetchone() != (2,):
            raise RuntimeError('Full synchronous writes are required')

    @classmethod
    def create(cls, destination, identity):
        identity = EmbeddingIdentity.from_dict(identity.to_dict())
        root = Path(destination).absolute()
        ensure_private_directory(root.parent)
        root.mkdir(mode=0o700)
        ensure_private_directory(root)
        path = root / 'memory.sqlite3'
        guard = acquire_writer_guard(root)
        connection = None
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, 'O_NOFOLLOW', 0), 0o600)
            os.close(descriptor)
            connection = cls._connect(path, read_only=False)
            connection.execute('BEGIN IMMEDIATE')
            for statement in SCHEMA.values():
                connection.execute(statement)
            connection.execute('INSERT INTO identity VALUES (1, ?)', (canonical_json(identity.to_dict()),))
            connection.execute(f'PRAGMA application_id = {APPLICATION_ID}')
            connection.execute(f'PRAGMA user_version = {SCHEMA_VERSION}')
            connection.commit()
            return cls(connection, identity, False, guard)
        except BaseException:
            try:
                if connection is not None:
                    connection.close()
            finally:
                guard.__exit__(None, None, None)
            raise

    @classmethod
    def open_read_only(cls, destination, identity):
        Path(destination).lstat()
        root = ensure_private_directory(destination)
        path = root / 'memory.sqlite3'
        connection = cls._connect(path, read_only=True)
        try:
            validate_database(connection, identity)
            return cls(connection, identity, True)
        except BaseException:
            connection.close()
            raise

    @classmethod
    def open_writable(cls, destination, identity, *, writer_timeout=0):
        identity = EmbeddingIdentity.from_dict(identity.to_dict())
        Path(destination).lstat()
        root = ensure_private_directory(destination)
        path = root / 'memory.sqlite3'
        validate_writable_header(path)
        guard = acquire_writer_guard(root, writer_timeout)
        connection = None
        try:
            validate_writable_header(path)
            connection = cls._connect(path, read_only=False, configure_writer=False)
            validate_database(connection, identity)
            cls._configure_writer(connection)
            return cls(connection, identity, False, guard)
        except BaseException:
            try:
                if connection is not None:
                    connection.close()
            finally:
                guard.__exit__(None, None, None)
            raise

    def insert_records(self, records):
        return self._write_records(records, upsert=False)

    def upsert_records(self, records):
        return self._write_records(records, upsert=True)

    def _write_records(self, records, *, upsert):
        if self.read_only:
            raise PermissionError('SQLite record store is read-only')
        validated = validate_records(records, self.identity)
        payloads = [(record.record_id, record.content, record.metadata_json,
                     struct.pack(f'<{self.identity.dimension}d', *record.embedding))
                    for record in validated]
        self._connection.execute('BEGIN IMMEDIATE')
        try:
            new_count = len(validated)
            if upsert:
                new_count = sum(self._connection.execute('SELECT 1 FROM records WHERE id = ?',
                                                        (record.record_id,)).fetchone() is None
                                for record in validated)
            if self.count() + new_count > MAX_RECORDS:
                raise ValueError('Record count exceeds scaffold limit')
            statement = 'INSERT INTO records VALUES (?, ?, ?, ?)'
            if upsert:
                statement += (' ON CONFLICT(id) DO UPDATE SET content = excluded.content,'
                              ' metadata = excluded.metadata, embedding = excluded.embedding')
            self._connection.executemany(statement, payloads)
            self._connection.commit()
        except BaseException:
            self._connection.rollback()
            raise
        return len(validated)

    def get_record(self, record_id):
        row = self._connection.execute('SELECT id, content, metadata, embedding FROM records WHERE id = ?', (record_id,)).fetchone()
        if row is None:
            return None
        return self._record_from_row(row)

    def count(self):
        return self._connection.execute('SELECT COUNT(*) FROM records').fetchone()[0]

    def search_vectors(self, embedding, *, limit=10, min_similarity=None,
                       min_importance=None, segment_types=None,
                       include_recycled=False):
        query = validate_vector(embedding, self.identity)
        if type(limit) is not int or not 1 <= limit <= MAX_RECORDS:
            raise ValueError('Invalid search limit')
        if type(include_recycled) is not bool:
            raise ValueError('Invalid recycled selection')
        if min_similarity is not None:
            if (self.identity.metric != 'cosine' or type(min_similarity) not in (int, float)
                    or not math.isfinite(min_similarity) or not -1 <= min_similarity <= 1):
                raise ValueError('Similarity threshold requires cosine and a finite value in [-1, 1]')
        if min_importance is not None and (type(min_importance) is not int or not 0 <= min_importance <= 100):
            raise ValueError('Invalid importance threshold')
        if segment_types is not None and (not isinstance(segment_types, (list, tuple))
                or len(segment_types) > 100 or any(not isinstance(value, str) for value in segment_types)):
            raise ValueError('Invalid segment types')
        prepared_query = prepare_cosine_vector(query) if self.identity.metric == 'cosine' else None
        def filtered_hits():
            for index, row in enumerate(cursor):
                if index >= MAX_RECORDS:
                    raise ValueError('Record count exceeds scaffold limit')
                candidate = self._search_cache.candidate(row, self.identity)
                record = candidate.record
                metadata = json.loads(record.metadata_json)
                if not include_recycled and metadata.get('recycled_at'):
                    continue
                if min_importance is not None and metadata.get('importance_score', -1) < min_importance:
                    continue
                if segment_types and metadata.get('segment_type') not in segment_types:
                    continue
                distance = (prepared_cosine_distance(prepared_query, candidate.cosine)
                            if prepared_query is not None
                            else vector_distance(query, record.embedding, self.identity.metric))
                yield VectorHit(record, distance)

        cursor = self._connection.execute('SELECT id, content, metadata, embedding FROM records')
        self._search_cache.begin_scan()
        completed = False
        try:
            matches = heapq.nsmallest(limit, filtered_hits(),
                                     key=lambda hit: (hit.distance, hit.record.record_id))
            completed = True
        finally:
            cursor.close()
            self._search_cache.end_scan(completed=completed)
        if min_similarity is not None:
            matches = [hit for hit in matches if 1.0 - hit.distance >= min_similarity]
        return matches

    def close(self):
        self._search_cache.clear()
        self._connection.close()
        if self._writer_guard is not None:
            self._writer_guard.__exit__(None, None, None)
            self._writer_guard = None

    def __enter__(self):
        return self

    def __exit__(self, *arguments):
        self.close()
