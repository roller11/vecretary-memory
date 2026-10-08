# SPDX-License-Identifier: MPL-2.0
"""Opt-in neutral record migration; never opens Chroma or switches a live backend."""

import hashlib
import json

from .private_storage import private_binary_reader, write_private_text
from .sqlite_records import MAX_MANIFEST_BYTES, EmbeddingIdentity, canonical_json, validate_records
from .sqlite_store import SQLiteRecordStore


FORMAT = 'vecretary-record-migration'
VERSION = 1
SCOPE = 'records_only'


def _checksum(value):
    return hashlib.sha256(canonical_json(value).encode('ascii')).hexdigest()


def _object_without_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f'Non-finite JSON constant: {value}')


def write_manifest(destination, identity, records):
    """Publish checksummed neutral records to a new owner-private file."""
    validated = validate_records(records, identity)
    payload = {'format': FORMAT, 'version': VERSION, 'scope': SCOPE,
               'embedding_identity': identity.to_dict(),
               'records': [record.to_dict() for record in validated]}
    envelope = {'payload': payload, 'sha256': _checksum(payload)}
    text = canonical_json(envelope) + '\n'
    if len(text.encode('ascii')) > MAX_MANIFEST_BYTES:
        raise ValueError('Manifest exceeds size limit')
    write_private_text(destination, text, replace=False)
    return {'scope': SCOPE, 'records': len(validated), 'sha256': envelope['sha256']}


def read_manifest(source, expected_identity):
    """Validate a bounded, trusted owner export without deserializing code."""
    with private_binary_reader(source) as stream:
        content = stream.read(MAX_MANIFEST_BYTES + 1)
    if len(content) > MAX_MANIFEST_BYTES:
        raise ValueError('Manifest exceeds size limit')
    try:
        envelope = json.loads(content, object_pairs_hook=_object_without_duplicates,
                              parse_constant=_reject_constant)
    except (UnicodeError, RecursionError) as error:
        raise ValueError('Invalid manifest JSON') from error
    if not isinstance(envelope, dict) or set(envelope) != {'payload', 'sha256'}:
        raise ValueError('Invalid manifest envelope')
    payload = envelope['payload']
    if not isinstance(payload, dict) or set(payload) != {'format', 'version', 'scope', 'embedding_identity', 'records'}:
        raise ValueError('Invalid manifest fields')
    if payload['format'] != FORMAT or type(payload['version']) is not int or payload['version'] != VERSION or payload['scope'] != SCOPE:
        raise ValueError('Unsupported migration format, version or scope')
    if envelope['sha256'] != _checksum(payload):
        raise ValueError('Manifest checksum mismatch')
    identity = EmbeddingIdentity.from_dict(payload['embedding_identity'])
    if identity != expected_identity:
        raise ValueError('Embedding identity mismatch')
    if not isinstance(payload['records'], list):
        raise ValueError('Manifest records must be a list')
    return validate_records(payload['records'], identity)


def import_manifest(source, destination, expected_identity, *, apply=False):
    """Dry-run by default; apply only to a new isolated record-store directory."""
    if type(apply) is not bool:
        raise ValueError('apply must be a boolean')
    records = read_manifest(source, expected_identity)
    report = {'scope': SCOPE, 'records': len(records), 'applied': False,
              'runtime_ready': False, 'verified': False}
    if not apply:
        return report
    with SQLiteRecordStore.create(destination, expected_identity) as store:
        store.insert_records(records)
    with SQLiteRecordStore.open_read_only(destination, expected_identity) as store:
        if store.count() != len(records):
            raise ValueError('Imported record count mismatch')
        for record in records:
            restored = store.get_record(record.record_id)
            if restored != record:
                raise ValueError('Imported record verification failed')
    return {**report, 'applied': True, 'verified': True}
