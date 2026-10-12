# Offline SQLite queue review and disposition

This owner-operated preparation tool works only from a verified SQLite offline
snapshot. It does not open a live/default store, stop services, switch backends,
automatically convert legacy operations, replay a queue or delete source files.
Explicit reviewed actions can convert eligible legacy saves, score updates and
deletes to SQLite schema-3 operations; originals are archived unchanged and
deletes become recoverable recycle requests, never permanent purge.

## Why legacy files are not converted

Schema-1/2 captures do not carry the schema-3 database, embedding and active-clock
bindings. Their wall timestamps cannot reconstruct historical active-time
creation/pop anchors. Explicit conversion therefore binds each operation to the
snapshot's SQLite identity and current clock; it preserves the original timestamp
but cannot recreate elapsed active time before that snapshot. Legacy saves are
re-embedded by the configured SQLite provider during replay. Updates and recycle
requests must target existing records and pass current agent authority checks. A
private file does not grant owner authority.

When possible, have the owner drain valid legacy work through its original
supported runtime before preparing a stopped snapshot. That is a separately
authorized operation, not performed by this tool. Otherwise, retain the bytes
for explicit review and manual reconciliation rather than claiming a converted
operation succeeded.

## Inventory and review

Use an explicit private identity JSON file and an existing stopped-runtime
[offline snapshot](SQLITE_OFFLINE_BACKUP.md). Inventory returns a private JSON
review plan; nothing is written by the inventory operation:

```sh
python -m instinct.sqlite_queue_review --snapshot /absolute/private-snapshot --identity /absolute/private-identity.json
```

The plan includes the entire snapshot fingerprint and each queue filename,
SHA-256 checksum, classification, operation kind and action. It excludes memory
content, titles, target IDs and reasoning; filenames can still be sensitive. Keep plans private.
Every action defaults to `retain`. Classifications are:

- `legacy_pending`: schema-1/2 data that passes the current ordinary-agent codec.
- `invalid_pending`: malformed, unsupported or authority-invalid data.
- `sqlite_binding_mismatch`: schema-3 data whose model/database/clock does not
  match the snapshot, including future or missing active-time anchors.
- `sqlite_pending`: schema-3 data passing static identity/clock checks only.
  Provider behavior, target availability and replay authority still require the
  normal runtime checks; this label does not promise successful replay.

The owner may change an individually reviewed `action` from `retain` to
`archive`, `convert_save`, `convert_score_update` or `convert_soft_recycle`, with
the conversion action matching the reported legacy operation. Conversion
preserves the original queued file in `archive/` and adds a schema-3 request to
`runtime/fallback_pending/`; it does not apply that request. No other review field
can change. Every file must appear once, in the
original order, with its current classification and digest. A stale plan,
changed snapshot, missing entry, unknown action or substituted file is refused.
Checksums detect changes; they do not authenticate an owner or trusted exporter.

## Dry-run and explicit application

Save the approved plan in a private JSON file, then request a dry run against a
new absolute destination outside the snapshot:

```sh
python -m instinct.sqlite_queue_review --snapshot /absolute/private-snapshot --identity /absolute/private-identity.json --review /absolute/private-review.json --destination /absolute/new-review-package
```

Only the additional `--apply` flag creates a candidate package. The API
equivalents are `inventory_snapshot` and `prepare_reviewed_restore`; application
requires `apply=True`. These are trusted offline owner interfaces, not MCP tools
or dashboard endpoints. Obtain explicit owner approval before using real
private snapshots or deciding the disposition of real pending work.

The new package contains:

- `runtime/`: byte-preserved database and recognized sidecars, plus retained
  pending files and any explicitly converted operations. No records,
  embeddings, scores, timers, pop anchors or receipts are recalculated. Runtime
  lock files are not copied.
- `archive/fallback_pending/`: exact bytes of individually approved archived
  files, including malformed data and originals of converted legacy deletes. It
  is outside the replay directory.
- `QUEUE_REVIEW.json`: snapshot-bound decisions and counts; always reports
  `runtime_ready=false`, even with no retained blocked files.

Archiving explicitly omits that pending operation from the new runtime queue; it
does not apply the operation, mark the original capture synced, or delete it.
Converted operations remain pending for normal owner replay, which rechecks guards
and provider availability. Converted deletes only recycle; permanent purge is
never generated. Retained legacy or invalid files remain visibly blocked in the
report. There is no default archive/conversion decision or inferred historical
active-time mapping.

## Retention, recovery and limits

The source runtime and source snapshot remain unchanged. Publication stages a
new private package before a same-parent rename. A publication failure can leave
an isolated staging directory; do not adopt it or treat it as a live runtime.
This does not certify every crash/power-loss window or multi-file recovery path.

Keep the original snapshot, the entire review package, its archive and the plan
until the owner has reconciled pending work. Subsequent offline backups target
`runtime/` only and do not include the sibling archive; retain that archive
separately. Never blindly reinsert archived captures or replay both old and new
runtimes: receipt/operation history requires explicit review.

Queue file/count/byte limits and owner-private file/directory checks apply.
Unknown snapshot state, integrity changes and unsafe paths fail closed. Pending
and archived bytes remain plaintext, not encrypted or securely erased. Same-user
malicious programs are outside the cooperative storage threat boundary.

The package uses only the SQLite runtime. This tool is not live-cutover permission,
a runtime switch or a claim that dependency advisories are resolved.
See [pending writes](SQLITE_PENDING_WRITES.md) and [owner purge](SQLITE_PURGE.md).
