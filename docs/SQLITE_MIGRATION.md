# SQLite migration

The migration helpers provide an explicit offline records-only bridge and a
staged records-plus-runtime-state bundle. They do not open or construct a source
database client, start a server, switch a live runtime, or declare a destination
ready for service.

## Export a stable source copy

Prepare an owner-controlled, stopped copy of the old collection. Never point a
legacy client at the original/live store for this operation. The source must not
change during export. Use a separately reviewed legacy environment to obtain an
already-open collection object and pass the exact embedding identity used by its
stored vectors:

```python
from pathlib import Path

from instinct.sqlite_migration import export_collection_manifest
from instinct.sqlite_records import EmbeddingIdentity

identity = EmbeddingIdentity(
    provider="ollama",
    model="nomic-embed-text",
    dimension=768,
    metric="cosine",
)
report = export_collection_manifest(
    Path("/absolute/private/new-records-export.json"),
    identity,
    already_open_collection,
)
```

The output path must be new and kept in a protected local directory. Export is
bounded to 10,000 records and 64 MiB, reads in pages, checks the collection count
before and after, and writes a checksummed neutral JSON manifest only after
validation succeeds.
Vectors are copied as stored; this step does not re-embed content. Record IDs,
documents, metadata and lifecycle fields are validated before publication.

The count check is not a database snapshot. It cannot detect every same-count
concurrent edit, which is why a stopped, unchanged copy is required. The
checksum detects accidental changes; it does not authenticate the exporter or
prove the source is trustworthy. Do not export untrusted databases.

## Stage records and runtime state together

For an end-to-end staged migration, pass the opened collection from the same
stopped source copy and its runtime directory to `export_runtime_bundle`. This
bundles the validated records with the recognized session clock, nudge state,
dormancy configuration, and pending JSON files. The exporter reads only these
known sidecars and `fallback_pending`; it does not copy the legacy database
directory or other files. Unknown entries inside the pending queue fail closed.

```python
from pathlib import Path

from instinct.sqlite_migration import export_runtime_bundle, import_runtime_bundle
from instinct.sqlite_records import EmbeddingIdentity

identity = EmbeddingIdentity(
    provider="ollama",
    model="nomic-embed-text",
    dimension=768,
    metric="cosine",
)
bundle = Path("/absolute/private/migration-bundle")
runtime_copy = Path("/absolute/private/stopped-source-copy")
destination = Path("/absolute/private/new-sqlite-runtime")

export_runtime_bundle(bundle, identity, already_open_collection, runtime_copy)
preview = import_runtime_bundle(bundle, destination, identity)
assert preview["runtime_ready"] is False
```

The bundle includes checksums and is published only after the records and state
validate. Legacy record metadata is converted to the Instinct storage shape
needed by the SQLite adapter; unsupported records reject the export rather than
being silently dropped. A checksum is an integrity check, not a signature or a
trust guarantee. Keep the bundle private because it contains memory content and
possibly queued writes.

After reviewing the dry-run report, apply only to a new isolated directory:

```python
result = import_runtime_bundle(bundle, destination, identity, apply=True)
assert result["verified"] is True
assert result["runtime_ready"] is False
```

Apply builds the database and sidecars in a sibling staging directory, verifies
record readback and exact sidecar/queue bytes, then atomically publishes the new
directory. Pending files are preserved byte-for-byte and classified for owner
review; none are acknowledged or replayed by the importer. Invalid and legacy
queue entries are retained rather than discarded. The separate queue-review flow
can archive or explicitly convert supported legacy operations. Do not direct a
running service at the staged directory: `runtime_ready` deliberately remains false
until queue disposition, restart/clock/decay/pop behavior, and rollback acceptance
pass separately.

## Validate and import into a new SQLite directory

The importer is dry-run by default. Review the report and identity, then use
`apply=True` only with a new isolated directory. It refuses existing destinations
and verifies every imported record by reading it back:

```python
from pathlib import Path

from instinct.sqlite_migration import import_manifest

preview = import_manifest(
    Path("/absolute/private/new-records-export.json"),
    Path("/absolute/new-runtime/sqlite"),
    identity,
)
assert preview["applied"] is False
```

Applying the import initializes only the records database; the report continues
to state `runtime_ready=False`. Do not point the MCP service at this directory
until all independent runtime files and invariants have been migrated and
verified.

## Preservation scope and module extension data

The records-only manifest preserves each record ID, document, complete JSON
metadata object and embedding. Runtime-compatible conversion validates and
normalizes known `ScoredSegment` fields, while unknown fields are carried through
the metadata object rather than discarded. This includes importance/initial
scores, reasoning, identifiers, mandatory scope/tags, decay and ephemeral state,
keyword history, active-time anchors, access/pop/promotion counters, recycle
metadata and namespaced `score_extensions`.

Runtime bundles reject malformed keyword-history or active-time-anchor JSON,
duplicate keys, non-finite numbers and unsupported shapes before publication or
import. These fields must not be silently reset by the runtime's legacy fallback
parser. Retain the source and obtain owner review for such records; the
records-only export can still preserve their original metadata for investigation.

The records-plus-runtime bundle also byte-preserves the recognized session,
nudge and dormancy configuration sidecars and pending queue files. Queue files
are never replayed or acknowledged during migration. It does not copy database
internals, lock files, archived backup snapshots or browser-local preferences;
those remain with the untouched source copy for separate review. The bundle is
still `runtime_ready=False` until queue dispositions and acceptance checks pass.

Future scoring modules should write their additive parameters under a stable
module namespace with `SQLiteSegmentAdapter.patch_score_extension`. Namespaced
JSON requires no SQLite column migration; core `importance_score` changes remain
transactional through `update_score`. A module that adds persistent sidecars must
also add a bounded validator and explicit migration/backup handling before that
state is claimed as migrated. Review unrecognized source files against the
preserved copy; never silently drop unknown sidecars or score data.

## Acceptance and rollback boundary

Migration acceptance is limited to the newly staged directory. Validate record
count, IDs, full metadata/extensions, embeddings and sidecar byte parity, then use the separate queue-review
workflow to classify SQLite-bound, mismatched, legacy, and malformed queue
files. Every queue requires an explicit reviewed disposition; migration itself
does not replay or delete any queue entry. The review tool can convert eligible
legacy saves, score updates and deletes into schema-3 pending operations while
archiving original bytes. Conversion binds to the snapshot clock; historical
active time cannot be reconstructed, and legacy saves are re-embedded on replay.
Validate timer-driven mandatory renewal, active-time decay, pop/access state,
recycle/restore, score-extension updates and restart behavior against
synthetic records before considering any later cutover.

The importer builds in a temporary sibling directory and publishes only after
verification. Injected failures during database construction, readback
verification, or final directory publication must leave the final destination
absent and preserve the source copy and migration bundle. This is staged-import
rollback acceptance only: it does not provide an automatic rollback for a live
service, prove power-loss durability on every filesystem, or authorize changing
the configured runtime. Keep the old runtime untouched until a separately
approved migration plan includes verified backups and an owner-operated
recovery procedure.
