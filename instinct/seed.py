# SPDX-License-Identifier: MPL-2.0
"""Explicit owner bootstrap for a new, dedicated Instinct installation."""

import argparse
from datetime import datetime
from importlib.resources import files
import json
import os
from pathlib import Path

from .models import MANDATORY_TYPES, ScoredSegment, Segment, SegmentType


MARKER_NAME = '.instinct-starter.json'
SEED_TIMESTAMP = datetime(2000, 1, 1)


def load_starter_memories():
    """Validate the bundled synthetic manifest before touching any store."""
    manifest = json.loads(files('instinct').joinpath('starter_memories.json').read_text(encoding='utf-8'))
    if not isinstance(manifest, dict) or type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1 or not isinstance(manifest.get('memories'), list):
        raise ValueError('Invalid starter manifest schema.')
    if not manifest['memories']:
        raise ValueError('Starter manifest is empty.')
    seen = set()
    memories = []
    for record in manifest['memories']:
        if not isinstance(record, dict):
            raise ValueError('Starter records must be objects.')
        template_id = record.get('template_id')
        if not isinstance(template_id, str) or not template_id.startswith('instinct.public.') or template_id in seen:
            raise ValueError('Starter template IDs must be unique public identifiers.')
        seen.add(template_id)
        if any(not isinstance(record.get(key), str) or not record[key].strip() for key in ('title', 'content')):
            raise ValueError('Starter title and content must be nonempty strings.')
        if type(record.get('importance_score')) is not int or record['importance_score'] != 100:
            raise ValueError('Starter memories must have score 100.')
        selected_type = SegmentType(record.get('segment_type'))
        if selected_type not in MANDATORY_TYPES:
            raise ValueError('Starter memories require mandatory types.')
        memories.append(ScoredSegment(
            segment=Segment(
                content=record['content'], segment_type=selected_type,
                timestamp=SEED_TIMESTAMP, source='instinct-starter',
                metadata={'starter_template_id': template_id, 'starter_schema_version': 1, 'mandatory_scope': 'always'},
            ),
            importance_score=100, concept_title=record['title'],
            reasoning='Owner-reviewed synthetic starter base',
        ))
    return memories


def seed_memories(store, memories, add_missing=True):
    """Create missing starter records, preserving existing owner edits and scores."""
    if getattr(store, '_read_only', False):
        raise ValueError('Starter seeding requires an owner-held writable store.')
    planned = []
    for memory in memories:
        existing = store.get(memory.id)
        if existing is not None and existing.segment.metadata.get('starter_template_id') != memory.segment.metadata['starter_template_id']:
            raise ValueError('Starter ID collision; no records have been changed.')
        planned.append((memory, existing is not None))
    created = []
    preserved = []
    missing = []
    for memory, exists in planned:
        if exists:
            preserved.append(memory.id)
        elif add_missing:
            store.add(memory)
            created.append(memory.id)
        else:
            missing.append(memory.id)
    return {'created': created, 'preserved': preserved, 'preserved_missing': missing}


def check_target(target, marker, expected):
    """Refuse unrelated stores and unrecognized or changed bootstrap manifests."""
    if not target.is_absolute():
        raise ValueError('Use an explicit absolute data directory.')
    if target.is_symlink() or marker.is_symlink():
        raise ValueError('Bootstrap target and marker must not be symlinks.')
    if marker.exists():
        from .private_storage import private_binary_reader
        with private_binary_reader(marker) as source:
            payload = json.load(source)
        if not isinstance(payload, dict):
            raise ValueError('Invalid starter marker.')
        state = payload.pop('state', None)
        if payload != expected or state not in ('initializing', 'complete'):
            raise ValueError('Starter marker differs; review an explicit migration rather than reseeding.')
        return state
    elif target.exists():
        if not target.is_dir():
            raise ValueError('Data path is not a directory.')
        allowed = {'.writer.lock'}
        if any(entry.name not in allowed for entry in target.iterdir()):
            raise ValueError('Refusing to seed a nonempty, unrecognized directory or live store.')


def write_marker(target, marker, payload):
    """Atomically reserve/complete bootstrap under the owner-held writer lock."""
    from .private_storage import write_private_text
    write_private_text(marker, json.dumps(payload, indent=2) + '\n')


def apply_starter(target, memories):
    """Lock and bootstrap only a fresh directory or a recognized starter install."""
    from .chroma_writer_lock import acquire_writer_lock, release_writer_lock
    from .store import InstinctStore

    target = Path(target)
    marker = target / MARKER_NAME
    expected = {'schema_version': 1, 'templates': {memory.segment.metadata['starter_template_id']: memory.id for memory in memories}}
    check_target(target, marker, expected)
    from .private_storage import ensure_private_directory
    ensure_private_directory(target)
    handle = acquire_writer_lock(str(target), timeout=0)
    try:
        state = check_target(target, marker, expected)
        if not marker.exists():
            write_marker(target, marker, {**expected, 'state': 'initializing'})
        store = InstinctStore(persist_dir=str(target))
        result = seed_memories(store, memories, add_missing=state != 'complete')
        if state != 'complete':
            write_marker(target, marker, {**expected, 'state': 'complete'})
        return result
    finally:
        release_writer_lock(handle)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Preview or explicitly apply owner-reviewed synthetic starter memories')
    parser.add_argument('--data-dir', required=True, help='Absolute dedicated installation data directory')
    parser.add_argument('--apply', action='store_true', help='Write missing starter records; default is preview only')
    parser.add_argument('--confirm-starter-base', action='store_true', help='Confirm owner review of all bundled starter memories')
    arguments = parser.parse_args(argv)
    if arguments.apply and not arguments.confirm_starter_base:
        parser.error('--apply requires --confirm-starter-base after owner review')
    try:
        memories = load_starter_memories()
        target = Path(arguments.data_dir).expanduser()
        preview = [{
            'template_id': memory.segment.metadata['starter_template_id'], 'id': memory.id,
            'title': memory.concept_title, 'content': memory.segment.content,
            'segment_type': memory.segment.segment_type.value, 'importance_score': 100,
        } for memory in memories]
        expected = {'schema_version': 1, 'templates': {memory.segment.metadata['starter_template_id']: memory.id for memory in memories}}
        check_target(target, target / MARKER_NAME, expected)
        result = apply_starter(target, memories) if arguments.apply else {'preview': preview}
        print(json.dumps({'data_dir': str(target), 'applied': arguments.apply, **result}, indent=2))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f'Starter setup failed: {error}\n')


if __name__ == '__main__':
    main()
