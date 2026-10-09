# Experimental SQLite single-record purge

This is an explicit owner-only API capability for an isolated SQLite runtime,
not a live migration or default behavior. Permanent agent deletion, bulk purge, reset,
automatic handoff and legacy queue conversion remain disabled.

## Enable and confirm

Add `--enable-purge` to the existing experimental SQLite owner launch. It is
false by default and incompatible with `--read-only`. The service constructor
equivalent is `enable_purge=True`. Health reports `purge_enabled`.

Use the existing authenticated dashboard API to recycle the selected record
first. Then POST `/api/purge` with exactly this JSON shape:

```json
{"id": "selected-record-id", "confirm_id": "selected-record-id"}
```

The existing Host/Origin/fetch-site checks, per-launch capability token and
bounded JSON parsing still apply. The ID must be a nonempty bounded string;
confirmation must match exactly. Active records are refused. A missing record
returns `purged: false`; a committed deletion returns `purged: true`.

This slice does not adapt the legacy dashboard purge buttons to send the extra
confirmation field. Those buttons fail closed instead of deleting a record.
The MCP `instinct_delete_memory` tool only moves eligible records into the
recoverable recycle bin; it cannot invoke permanent purge. An opted-in observer
may queue that soft-recycle operation, but pending replay never grants purge
authority. Owner-only purge still requires the explicit confirmation above.

## Queue and transaction safety

Purge holds the existing pending capture guard for the queue inspection and
database transaction. Any supported, legacy or malformed pending JSON file
blocks purge, even if unrelated to the selected record. This prevents removing
receipt-bearing records while unacknowledged operations could still replay.
Purge never drains or modifies pending files itself; review legacy/invalid files
or drain the supported queue through the explicit replay policy first.

Recycle validation and single-record deletion share a SQLite transaction. A
failure rolls back; only the selected row is deleted. No embedding provider,
identifier cache rewrite, activity tick, score update or pop reset is involved.
Other memories and sidecars remain unchanged. Capture-guard contention or
unsafe queue/storage state refuses the operation.

A database error or lost response is unconfirmed, not permission to blindly
retry. Inspect committed records first. Same-user malicious programs are outside
the cooperative guard/capability threat boundary.

## Recovery and acceptance limits

Purge permanently removes a record from the current logical database. Restore
from a separately retained pre-purge offline snapshot if recovery is needed.
Post-purge snapshots retain its absence. This is not secure erasure: old backups,
SQLite free pages, filesystem snapshots and storage media can retain bytes.
Backups must remain private and follow the owner's retention policy.

Synthetic tests cover opt-in/default refusal, observer refusal, exact confirmation,
authenticated HTTP boundaries, recycle-first policy, record/clock isolation,
pending-file and guard refusal, transaction rollback, restart, and pre/post-purge
backup restoration. They do not authorize deleting real private memories or
certify every power-loss window. `runtime_ready=false` remains; the SQLite
service does not expose bulk reset or legacy backend selection.

See [service integration](SQLITE_SERVICE_INTEGRATION.md),
[pending writes](SQLITE_PENDING_WRITES.md) and
[offline backup](SQLITE_OFFLINE_BACKUP.md).
