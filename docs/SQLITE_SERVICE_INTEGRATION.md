# SQLite MCP/dashboard integration

The published alpha's `vecretary-mcp` entry point selects the SQLite service.
It requires an explicitly initialized, new SQLite runtime root and never
implicitly migrates, repairs, seeds or switches a legacy store. The integration
tests use synthetic data only. Passing them does not authorize a live migration.

`vecretary-init` creates an empty new root with an explicit Ollama/cosine
identity; initialization does not probe the provider. Existing roots are never
overwritten. An interrupted initialization may leave a partial root: review it
and choose another new path rather than attempting implicit repair. See the
[SQLite alpha README](sqlite-release/README.md) and the
[offline migration and limits](SQLITE_MIGRATION.md).

## Ownership and request routing

`SQLiteService` opens an existing recognized database inside an explicit existing
runtime root. One dedicated executor thread owns the connection, runtime sidecar
guard and complete MCP/HTTP operations. The dispatcher never sends that connection
to arbitrary executor workers. At most 16 requests can be in flight or queued.
Shutdown drains accepted operations before closing the owning connection.

Stdio MCP and the browser dashboard can share this owner in one process. Each
stdio connection has its own mandatory-renewal lifecycle. There is no new shared
writer socket protocol or automatic writer takeover. Other owners of the same
runtime root are rejected. Read-only observers acquire neither writer guard.

Request-local bindings select the adapter, dormancy configuration and default
activity clock without changing legacy global stores or environment configuration.
MCP calls retain the existing activity/session tick, turn tick, mandatory renewal,
context nudge checks, write authority and retrieval/pop pipeline. SQLite timer or
nudge failure aborts dispatch rather than silently permitting a new memory write.
Metadata and timer sidecars still commit separately: this is not an atomic
cross-file operation or an exactly-once protocol.

## Available functionality

- MCP context, mandatory reads, semantic search, individual reads, ordinary saves
  and score updates reuse the existing tool handlers and input limits.
- Agent saves and updates retain the score-92 ceiling and protected-target checks.
  Owner dashboard writes retain the existing owner-specific upper-score behavior.
- The existing dashboard pages provide list/detail, search, model editing,
  recycle/restore, access tracking, pop actions, decay controls/previews and nudge
  controls through an explicit route allowlist.
- Bounded statistics and backend-neutral health/mode APIs support the existing
  browser status widgets without opening a separate storage backend or writer lock.
- Both decay periods remain configurable through the existing dashboard config
  schema. Scoring formulas, score ranges and floor values are not changed here.
- Heartbeats use the selected runtime's active-time clock without legacy takeover
  or watchdog side effects. Idle wall time is not charged on owner restart.
- Backend-neutral readiness reports store access, binding presence and explicit
  experimental status. Binding presence is not a live embedding-provider probe.

Exact SQLite vector search uses a connection-local validated-record and cosine
preparation cache, bounded to 512 records and a conservative 32 MiB retained-byte
budget. Every search still scans the current SQL rows: only an exact row/embedding
identity match can reuse validated data. Changed records are revalidated even
when they would be filtered out or miss the top results. Owner edits, observer
reads after commits, rollback, recycle/restore and purge do not use stale cached
rows. The cache is cleared on close and is never persisted or included in backups.
Admission is scan-resistant: existing validated candidates are retained instead
of being evicted for every capacity miss. Unadmitted rows are fully validated and
prepared on every scan. Completed scans reclaim cached IDs no longer present;
failed scans do not discard unvisited valid candidates. Freed slots may be filled
on the following scan. The record and accounted-byte limits are unchanged.

Record validation still checks every field, metadata depth, finite vector value
and dimension. A conservative canonical-JSON upper bound (64 bytes per normalized
finite float, including delimiter allowance) avoids redundant vector serialization
only when the whole record is safely below its existing size limit. Near-boundary
records use the original exact canonical-size check. Manifest byte quotas still
use exact serialization. No hash-only validity check or weaker corruption handling
is used. Cosine products retain the same operation order and double precision.
No dependency, schema, search-score, clock, pop or ordering changes are introduced.
Cold scans still validate every record; larger scans exceed the cache bounds and
must validate the unadmitted remainder. This is not an index or a scalability guarantee.

### Namespaced score extensions

New modules can store additive scoring parameters with
`SQLiteSegmentAdapter.patch_score_extension(record_id, namespace, changes)`. The
method validates bounded JSON and merges the patch transactionally into the
record's `score_extensions[namespace]` metadata. Modules should use a stable
package/reverse-domain namespace and version their own payload. This avoids
module-specific SQL columns and survives metadata writes, score changes, backups
and migration. Extension factors do not change retrieval ordering by themselves;
modules that intentionally change the canonical importance score must call the
existing `update_score` contract and preserve owner/agent write policy.

### Synthetic performance checks

The source checkout includes an opt-in, stdlib-only
[benchmark runner](../tests/run_sqlite_benchmark.py). It creates new temporary
synthetic databases, batches inserts and deletes only its own temporary fixtures.
It never selects a live backend or opens an existing memory store.

```sh
python -m tests.run_sqlite_benchmark --confirm-synthetic-benchmark --output benchmark-results.json
```

Defaults cover 230, 512, 1,000, 5,000 and 10,000 records at 768 dimensions, with
20 samples per scenario. The full default run can take tens of minutes. Options `--sizes`,
`--dimension`, `--samples` and `--seed` support smaller reproducible runs.
The output must be new; existing reports are not overwritten.

Measurements separate cold, repeated varied-query, filtered, post-metadata-write
and post-vector-upsert scans from write latency. Every measured search is checked
against uncached cosine arithmetic and independently sorted exact top-10 results;
metadata freshness, cache bounds and read-only reopen are also checked. Reference
calculation is outside the timed region. P95 is omitted below 20 samples; one run
with 20 samples is still only exploratory evidence, not a service-level guarantee.

The 10-percent metadata workload uses one fully synchronized transaction per
record, not one bulk transaction. Cache accounted bytes are not total process RSS.
These sequential record-layer checks exclude embedding generation, MCP/dashboard
transport, active-clock/pop dispatch, concurrency and crash recovery. Those remain
separate integration requirements. At most 512 records benefit from reuse;
larger corpora still require substantial uncached validation and preparation.

The pre-optimization October 7, 2026 synthetic WSL/Python 3.12 run at 768 dimensions measured
median repeated scans of about 15 ms at 230 records, 34 ms at 512, 995 ms at
1,000, 5.1 seconds at 5,000 and 10.5 seconds at 10,000. A shorter Windows run
confirmed the same cache-capacity cliff. This historical result motivated the
scan-resistant admission and bounded-size validation above. These are host-specific
exploratory measurements, not SQLite-engine limits. Repeat workloads after changes
and validate actual interactive latency before approving a larger-store live cutover.

Repeating the same 20-sample synthetic protocol after these optimizations measured
median repeated scans of 10.6 ms at 230 records, 23.6 ms at 512, 158.6 ms at
1,000, 1.30 seconds at 5,000 and 2.80 seconds at 10,000. All exact-reference
checks passed, and the 512-record/32 MiB accounted-byte bounds were preserved.
The shorter Windows confirmation measured 218.5 ms at 1,000 records. Large-corpus
scans remain linear and can still take seconds; this is not approval for a live
switch or a guarantee of responsive large-store dashboard/MCP operation.

Dashboard writes retain Host/Origin/fetch-site checks, per-launch capability tokens,
bounded JSON, security headers and CSP. The server binds only `127.0.0.1`.
These controls do not protect against a malicious local user or process. Slow
requests can temporarily occupy the serialized owner; existing request timeouts
and the bounded backlog limit this, but do not provide production throughput.

## Recycle and permanent deletion

`instinct_delete_memory` moves an eligible record into SQLite's recoverable recycle
bin. It preserves record metadata and vectors and refuses protected owner records.
Read-only observers can capture this soft-recycle request only when queue capture
is explicitly enabled; replay records a receipt transactionally. Permanent purge,
admin reset, dashboard backup/recovery and writer handoff remain unavailable or
separately gated. Owner-only single recycled-record purge is available with
[explicit opt-in and exact ID confirmation](SQLITE_PURGE.md); it remains disabled
by default. Unsupported dashboard endpoints return 501.
The page displays an experimental-capability notice; some legacy buttons remain
visible and are safely refused by the server rather than presented as working.

Default observer MCP writes and dashboard mutations fail without creating fallback files.
Observers do not advance turn/session state. Mandatory context reads remain blocked
when that connection needs writable renewal. The default SQLite path never examines
pending queues. Explicit schema-3 replay classifies but never acknowledges legacy
records. A cancelled/disconnected caller does not cancel
an accepted owner operation; confirm persistence before retrying a write.

Ordinary MCP saves, score updates and soft-recycle requests have [explicit opt-in
pending capture and receipt-based replay](SQLITE_PENDING_WRITES.md). Use `--read-only --queue-writes`
for a capture observer or `--replay-pending` for an owner, never both. Queueing is
not a database commit, and this capability does not add automatic writer handoff.

## Explicit launch

Prepare an isolated records database and any separately reviewed sidecars first.
The integration does not migrate, repair, seed or create a database implicitly.
The identity file is a JSON object with `provider`, `model`, `dimension` and `metric`
matching the existing database exactly. Only cosine text search is supported.

```sh
python -m instinct.sqlite_service \
  --database-dir /absolute/isolated-runtime/sqlite \
  --data-dir /absolute/isolated-runtime \
  --identity /absolute/embedding-identity.json \
  --provider-config /absolute/local-provider.json \
  --dashboard-port 8767
```

This command starts stdio MCP plus the dashboard. The dashboard URL is printed to
stderr so stdout remains reserved for MCP. Configure the MCP client to own this
process; stopping it closes both services. Do not launch the legacy dashboard
against the SQLite root. Passing `--read-only` explicitly selects an observer.

The CLI currently accepts only one explicitly configured Ollama provider. Its
model must already be available locally and match the identity. Loopback URL and
provider/model/metric bindings are checked before embeddings are requested; no
provider fallback or model-download command is used. The provider config uses the
existing format, for example:

```json
{
  "providers": ["ollama"],
  "ollama": {
    "model_name": "your-already-installed-model",
    "base_url": "http://127.0.0.1:11434",
    "timeout": 30
  }
}
```

Without `--provider-config`, existing-record metadata operations and reads remain
available, but text search/new or changed content fail explicitly. Other providers
require reviewed direct API bindings; synthetic injected providers are for tests
only. No production provider/model is certified by synthetic acceptance.

## Acceptance and remaining gates

An explicit, stopped single-collection root now has an experimental
[offline backup/restore API](./SQLITE_OFFLINE_BACKUP.md). It preserves recognized
sidecars and pending bytes without replay. Dashboard backup endpoints remain
disabled; fresh live migration and multi-file crash recovery remain separate gates.

Tests exercise real MCP initialization/tool transport, actual loopback HTTP,
shared owner-thread routing, input/authority refusals, independent mandatory
lifecycles, observer immutability, cancellation, queue isolation, bounded request
capacity, and active-clock/access/vector persistence across restart. Existing
active-time and both pop suites remain regression requirements.

Before a live switch: obtain explicit owner approval for private-data access and
service changes; validate staged full-state/sidecar migration and backup/restore
policy; review [legacy queue disposition](SQLITE_QUEUE_REVIEW.md) if required;
validate actual embedding-provider behavior and performance on representative
data. Legacy storage and process-control modules are removed from this profile.
The offline migration exporter accepts an already-open collection object but
does not construct a legacy client or import a legacy database file. Keep any
legacy source environment separate from the SQLite runtime.
