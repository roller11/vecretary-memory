# Offline SQLite queue review and disposition

This owner-operated preparation tool works only from a verified SQLite offline
snapshot. It does not read Chroma, access a default store, stop services, switch
backends, convert legacy operations, replay a queue or delete source files.

## Why legacy files are not converted

Schema-1/2 captures do not carry the schema-3 database, embedding and active-clock
bindings. Their wall timestamps cannot reconstruct active-time creation/pop
anchors. Guessing the missing state could age or reset a memory incorrectly.
Legacy deletes also lack the newer replay/receipt policy. A private file does
not grant owner authority or override ordinary-agent score/type guards.

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
SHA-256 checksum, classification and action. It excludes memory content, titles,
target IDs and reasoning; filenames can still be sensitive. Keep plans private.
Every action defaults to `retain`. Classifications are:

- `legacy_pending`: schema-1/2 data that passes the current ordinary-agent codec.
- `invalid_pending`: malformed, unsupported or authority-invalid data.
- `sqlite_binding_mismatch`: schema-3 data whose model/database/clock does not
  match the snapshot, including future or missing active-time anchors.
- `sqlite_pending`: schema-3 data passing static identity/clock checks only.
  Provider behavior, target availability and replay authority still require the
  normal runtime checks; this label does not promise successful replay.

The owner may change an individually reviewed `action` from `retain` to
`archive`. No other field can change. Every file must appear once, in the
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
  pending files. No records, embeddings, scores, timers, pop anchors or receipts
  are recalculated. Runtime lock files are not copied.
- `archive/fallback_pending/`: exact bytes of individually approved archived
  files, including malformed data. It is outside the replay directory.
- `QUEUE_REVIEW.json`: snapshot-bound decisions and counts; always reports
  `runtime_ready=false`, even with no retained blocked files.

Archiving explicitly omits that pending operation from the new runtime queue;
it does not apply the operation, mark the original capture synced, or delete it.
Review desired memory effects before approving archival. Retained legacy or
invalid files remain visibly blocked in the report. There is no default archive
decision and no inferred active-time mapping.

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

The default package still uses Chroma. This tool is not live-cutover permission,
a backend selection change or a claim that dependency advisories are resolved.
See [pending writes](SQLITE_PENDING_WRITES.md) and [owner purge](SQLITE_PURGE.md).
