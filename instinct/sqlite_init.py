# SPDX-License-Identifier: MPL-2.0
"""Explicit empty SQLite runtime creation; never migrate or overwrite a root."""

import argparse
import json
import os
from pathlib import Path

from .private_storage import ensure_private_directory
from .sqlite_records import EmbeddingIdentity, canonical_json
from .sqlite_store import SQLiteRecordStore


def initialize(data_dir, identity):
    identity = EmbeddingIdentity.from_dict(identity.to_dict())
    if identity.provider != 'ollama' or identity.metric != 'cosine':
        raise ValueError('Fresh CLI runtimes require an explicit Ollama/cosine identity')
    root = Path(data_dir).expanduser()
    if '..' in root.parts:
        raise ValueError('Runtime paths must not contain parent traversal')
    root = root.absolute()
    ensure_private_directory(root.parent)
    root.mkdir(mode=0o700)
    ensure_private_directory(root)
    with SQLiteRecordStore.create(root / 'sqlite', identity):
        pass
    identity_path = root / 'embedding-identity.json'
    descriptor = os.open(identity_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, 'w', encoding='ascii') as output:
        output.write(canonical_json(identity.to_dict()) + '\n')
    return root


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True, help='New, nonexistent runtime root')
    parser.add_argument('--identity', type=Path, required=True, help='Explicit embedding identity JSON')
    arguments = parser.parse_args()
    with arguments.identity.open('rb') as source:
        payload = source.read(4097)
    if len(payload) > 4096:
        parser.error('Identity file exceeds 4096 bytes')
    try:
        identity = EmbeddingIdentity.from_dict(json.loads(payload))
        root = initialize(arguments.data_dir, identity)
    except (ValueError, OSError, UnicodeError) as error:
        parser.error(str(error))
    print(json.dumps({'initialized': True, 'backend': 'sqlite', 'data_dir': str(root),
                      'records': 0, 'provider_probed': False, 'runtime_ready': False}))


if __name__ == '__main__':
    cli()
