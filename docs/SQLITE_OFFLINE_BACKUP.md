# Experimental SQLite offline backup

This is an opt-in API for a stopped, isolated, single-collection SQLite runtime.
It is not a live database migration command or a dashboard backup endpoint.
The default runtime remains unchanged and `runtime_ready` remains false.

A verified snapshot can feed the [offline queue-review tool](SQLITE_QUEUE_REVIEW.md).
It creates a new package with separate runtime and owner-approved archival
copies; it never modifies this snapshot. Subsequent backups target that package's
`runtime/` directory only. Retain its sibling archive and review plan separately.

## Supported state

- The recognized `sqlite/memory.sqlite3` database, including original vectors,
  metadata, active-time anchors, access counters and both pop mechanisms.
- Exact bytes of existing `session_state.json`, `nudge_state.json` and
  `dormancy_config.json`; missing files remain missing. Clock identity and the
  accumulated active-time value are not regenerated during backup/restore.
- Existing `fallback_pending/*.json`, preserved as opaque pending records.
  Nothing is replayed, acknowledged or deleted.

The runtime directory must contain only these entries, the database directory,
optional pending directory and recognized lock files. Unknown files, another
collection, temporary state and SQLite journals fail closed rather than being
silently omitted. Lock files are validated but not copied; they represent OS
ownership, not portable application state. JSON sidecars must be finite objects;
this is not certification of every legacy sidecar field's semantic validity.

## Explicit API

```python
from pathlib import Path
from instinct.sqlite_backup import snapshot_offline, restore_new
from instinct.sqlite_records import EmbeddingIdentity

identity = EmbeddingIdentity('ollama', 'your-installed-model', 768, 'cosine')
snapshot_offline(Path('/absolute/private/runtime'),
                 Path('/absolute/private/backup-new'), identity)
restore_new(Path('/absolute/private/backup-new'),
            Path('/absolute/private/restored-new'), identity)
```

Use existing owner-private parents and new absolute destinations outside the
source. The database must be at `runtime/sqlite/memory.sqlite3`. Stop all users
of that isolated root first; never use the legacy dashboard against it. The API
acquires the runtime and database writer guards with zero wait, and the pending
capture guard when present. It refuses contention without deleting locks.
Only cooperative, quiescent ownership is supported: a legacy queue replay or
an external program ignoring these guards must not run concurrently.

Files are bounded, privacy checked, hashed and staged in a private sibling
directory before publication. A second inventory detects source changes. A
manifest records format/schema, embedding identity, record count and checksums.
Restore verifies the inventory, recognized database schema/integrity and every
record before publishing. Existing destinations are refused. Destination parents
must be trusted and not changed concurrently by other programs.

Checksums detect accidental changes; they are not signatures, authenticity proof
or encryption. Snapshots contain private memories. Do not publish them. Failed
staging directories remain private for inspection and owner-directed cleanup.
File flushes and directory rename are not a tested power-loss durability protocol.

## Remaining gates

- Fresh, explicitly approved full-state migration from the live source, with
  stopped services and an inventory mapping for all excluded legacy artifacts.
- A multi-collection policy; this API never silently merges collections.
- Forced-crash recovery across separately committed sidecars and records.
- Reviewed hard deletion and legacy queue conversion if those capabilities ship.
  [Schema-3 queue replay](SQLITE_PENDING_WRITES.md) is now an explicit opt-in;
  backup/restore itself still never replays or acknowledges pending files.
- Explicit live cutover approval, rollback rehearsal and release packaging.

Synthetic round-trip and failure tests accompany the API. Private real-provider
acceptance is separate evidence; it does not certify arbitrary model identifiers,
large workloads, all crash windows or complete distribution independence from
native libraries.
