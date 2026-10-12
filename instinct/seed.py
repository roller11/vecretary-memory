# SPDX-License-Identifier: MPL-2.0
"""Owner-reviewed starter memories for explicitly configured store adapters."""

from datetime import datetime
from importlib.resources import files
import json

from .models import MANDATORY_TYPES, ScoredSegment, Segment, SegmentType


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


def apply_starter(store, memories):
    """Add only missing starter memories through the caller's SQLite adapter."""
    return seed_memories(store, memories)
