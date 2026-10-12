# Experimental SQLite pending writes

SQLite can now explicitly reuse the existing private pending-file capture codec,
quotas and guards for ordinary agent saves, score updates and recoverable recycle
requests. Both capture and
replay remain **off by default**. No live installation is changed by adding this
source capability; `runtime_ready=false` remains intentional.

## Roles and lifecycle

The existing SQLite service requires an explicit database directory, runtime root
and embedding identity as described in the [integration guide](SQLITE_SERVICE_INTEGRATION.md).

- Add `--read-only --queue-writes` for an observer that may capture MCP saves,
  score updates and soft-recycle requests into that root's `fallback_pending` directory. It does not change
  records, tick the activity timer, change nudge state or reset an owner's
  monotonic clock. A normal observer without this flag remains byte-immutable.
- Add `--replay-pending` for an owner. It replays at startup and before MCP calls
  or authenticated dashboard POST operations. Database and runtime guards must
  already have been acquired; contention never causes lock deletion or takeover.
- Capture and replay roles are mutually exclusive. This slice does not add
  automatic owner/observer promotion or writer handoff. An existing owner with
  replay enabled can consume captures on its next request. Otherwise an owner
  must be started explicitly after the previous writer releases its guards.
- Dashboard observer mutations still return read-only errors. Only MCP save,
  score-update and soft-recycle tools capture; there is no unauthenticated HTTP queue endpoint.

The equivalent API options are `SQLiteService(..., read_only=True,
queue_writes=True)` and `SQLiteService(..., read_only=False,
replay_pending=True)`. Owner text saves require the existing explicit document
embedding binding. Capture needs no provider: duplicate-title checks use existing
metadata. The initialized runtime clock must already have a valid identity.

## State and authority

New captures use pending schema 3, containing the original timestamp, ordinary
agent authority, exact embedding identity, root-relative database selection, and a snapshot of the
canonical activity clock. Original wall timestamps and captured active seconds
survive replay; waiting while the owner is active counts toward retained-memory
  age, while idle wall time does not. A queued score change sets its pop anchor to
  the original capture point, preserving access counts, initial score, other pop
  state and the existing search-score formula. A queued delete only sets recycle
  metadata; it cannot permanently purge a record.

Capture and replay enforce the 92 agent cap and protect mandatory types and
owner-only initial scores, including temporarily suppressed owner memories.
Save content/title guards are checked again during replay, but a valid capture
is not charged twice against the process-local high-priority rate limit.
Database/embedding/clock mismatches, future active-time anchors, missing/protected targets,
duplicate committed titles, embedding failures or invalid files are retained for
review. Queued files do not grant owner privileges merely because they are private.

Legacy schema 1/2 records are **not** automatically converted or replayed by this
SQLite protocol. An [offline review/staging tool](SQLITE_QUEUE_REVIEW.md) can retain
or archive them, or explicitly convert a reviewed save, score update or delete
into a schema-3 operation. Conversion binds to the snapshot identity and current
clock; it preserves original timestamps but cannot reconstruct historical active
time. Legacy saves are re-embedded on replay. Original queue bytes are archived,
and converted deletes only soft-recycle. The SQLite runtime accepts only schema-3
queue records, and the adapter advertises `pending_replay_ready=false` so
unsupported adapters cannot replay them.

## Commit and recovery

Each schema-3 envelope has a canonical content digest. A queued save gets a stable
`queued_` ID derived from that digest; score updates target the existing ID.
The operation and its digest receipt commit together in the same SQLite record
transaction. Only afterward is the pending file acknowledged by removing it.
If the process exits after commit but before acknowledgement, replay confirms the
receipt without re-embedding, duplicating a save, resetting pop/access anchors or
reapplying an old score over a newer owner edit. Receipts are metadata, not a new
database table or schema migration.

Replay holds the pending capture guard and processes at most 32 valid SQLite
records per pass in pending filename order. It stops at the first failed supported
operation, leaving subsequent operations pending. Unreviewed legacy/invalid files
are retained and counted separately. New MCP writes and dashboard mutations are
refused when supported pending records have not drained; reads and heartbeat can
continue. Health exposes enabled policies and last replay counts without payloads.

Receipts are bounded to 1,024 per record and are never automatically pruned.
Capacity exhaustion requires owner review. The existing pending-file count/byte
quotas and private permissions still apply. Captures are plaintext durable pending
files, not guaranteed database commits or encrypted storage.

A lost response remains ambiguous: inspect the pending file **and committed
records** before retrying. Retrying a capture can create another request envelope;
this is not an exactly-once client protocol. Trusted same-user programs must not
edit snapshots/queue files concurrently outside the cooperative guards.

## Acceptance scope

Synthetic tests cover real observer CLI/owner stdio transport, authenticated HTTP,
authority checks, provider/acknowledgement failures, bounded replay/receipts,
clock/pop continuity, backup/restore, and abrupt process exit in the specific
commit-before-acknowledgement window. This does not certify every multi-file crash
window, arbitrary model performance or power-loss directory durability.

An [explicit owner-only recycled-record purge API](SQLITE_PURGE.md) requires an
empty pending queue under the capture guard. Permanent agent deletion, bulk purge/admin
reset, legacy queue migration and live switching remain separate gates.
This SQLite package does not interpret or replay legacy queue records automatically.
