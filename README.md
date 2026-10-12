# Vecretary Memory - SQLite alpha

<p align="center"><img src="instinct/assets/vecretary-logo-full.png" width="220" alt="Vecretary"></p>

<p align="center">Local, durable, priority-aware memory for MCP-capable agents.</p>

The owner dashboard uses a charcoal and green theme in dark and light modes. Its
primary navigation surfaces Memories, Score helper, Decay and Recycle bin; a
prominent New memory action is separated from grouped secondary maintenance
tools. Large workspace panels keep creation, retrieval and detail areas distinct.
Static mobile-size previews (no live service status or memory data) are available
in [dark theme](docs/assets/dashboard-mobile-dark.png) and
[light theme](docs/assets/dashboard-mobile-light.png).

Vecretary provides local, persistent, priority-aware memory for MCP-capable
agents. Semantic recall finds related records; importance, active-time decay,
mandatory context and temporary pops keep useful context readily available.
The interface is agent/vendor agnostic: changing your chat model need not change
your memory store or embedding model. Agents must still be instructed to call
the tools; Vecretary cannot compel tool use or instruction compliance.

## Benefits and proposed uses

- Retain project decisions, constraints and preferences across sessions/agents.
- Recall research findings and identifiers alongside related vector results.
- Keep long-running task context prominent rather than treating all notes alike.
- Inspect and edit retained memories locally through a loopback dashboard.
- Control retention and make reviewed offline backups without a database server.

These are intended uses, not guarantees of recall, privacy or agent correctness.
This alpha is not approved for regulated, hostile or multi-tenant environments.

## Implemented features

- Embedded SQLite records and vectors; exact linear cosine scans, without an external vector-store dependency
  as an installed dependency. No database HTTP server is started.
- Structured content, titles, IDs, memory types, importance and lifecycle metadata.
- Future scoring modules can add namespaced JSON parameters without adding
  module-specific SQLite columns; metadata extensions survive score updates and migration.
- MCP context/mandatory retrieval, semantic search, individual reads, ordinary
  saves, eligible score updates and recoverable recycle requests. MCP recycling
  never permanently deletes a record.
- Existing identifier matching, result filtering, importance/modifier scoring,
  access tracking and both temporary-pop mechanisms. Search scoring is unchanged;
  it is not the proposed `importance + semantic / 2` formula.
- Active-session timer, turn/session tick, mandatory renewal and nudge scheduling.
  Scores 99-100 are surfaced; access lowers an elevated mandatory to 79 and its
  configured turn interval (five turns by default) restores it to 99. Idle time
  is not charged on restart.
  Tokens acknowledge retrieval, not compliance.
- Configurable initial-drop and later-decay periods, curves, floors and retention.
  Bands include 0-49, 50-79 and 80-89, whose default floor is 60; special upper
  profiles remain. Ordinary agent scores are capped at 92; mandatory/higher
  scores and protected-target changes remain owner-only.
- Owner dashboard list/search/edit, score controls, statistics, diagnostics,
  decay previews/configuration, pop actions, nudge, recycle and restore.
- One serialized owner serves stdio MCP and loopback HTTP, with bounded backlog;
  read-only observers do not acquire writer authority or advance the timer.
- Explicit opt-in pending-save/score/recycle capture with receipt-based replay;
  capture is not a committed write and does not provide automatic writer handoff.
- Offline backup/restore APIs, queue review and opt-in single-record recycled
  purge with exact ID confirmation. No implicit repair, migration or seeding.

MCP recycle requests preserve records and vectors and can be restored from the
dashboard. Permanent purge remains owner-only, separately gated and is not an MCP tool.

Unsupported dashboard reset, online backup and writer-handoff routes fail closed.
Database and timer/config/queue sidecars commit separately, not atomically.
See [integration](docs/SQLITE_SERVICE_INTEGRATION.md),
[offline migration and limits](docs/SQLITE_MIGRATION.md),
[pending writes](docs/SQLITE_PENDING_WRITES.md),
[offline backup](docs/SQLITE_OFFLINE_BACKUP.md) and [purge](docs/SQLITE_PURGE.md).

## MCP tools

| Tool | Purpose |
| --- | --- |
| `instinct_get_context` | Retrieve relevant, priority-ranked context. |
| `instinct_get_mandatory` | Retrieve mandatory standing context. |
| `instinct_search` | Search semantically with score and similarity filters. |
| `instinct_get` | Retrieve one memory by exact ID without result truncation. |
| `instinct_save_memory` | Store a new memory through the write guards. |
| `instinct_update_score` | Update importance through owner and mandatory-record protections. |
| `instinct_delete_memory` | Move an eligible memory to the recoverable recycle bin. |

## Fresh isolated setup

Reviewed release target: Linux x86_64, Python 3.11/3.12. Native Windows release
support remains unapproved. Install the reviewed wheel in a new virtual environment.
Use the accompanying `requirements-linux.txt` as installation constraints to
reproduce the reviewed runtime dependency versions; reassess advisories before
each release. It is not a hash-pinned artifact lock or a system-SQLite audit.
Bootstrap the reviewed installer/build tooling in that isolated environment:

```sh
python -m pip install --upgrade pip==26.2.1 setuptools==83.0.0
```

For the downloadable alpha, obtain the wheel and requirements-linux.txt from
the GitHub prerelease. Alternatively, build a wheel from the repository root:

```sh
python -m pip wheel . --no-deps --wheel-dir dist
```

Install that wheel with its constraints:

```sh
python -m pip install -c requirements-linux.txt dist/vecretary_memory-0.2.0a2-py3-none-any.whl
```

An already-installed local Ollama model is required for text operations; there
is no ONNX fallback, model download or implicit provider selection.

Create an identity file matching that model's real output dimensions:

```json
{"provider":"ollama","model":"your-installed-model","dimension":768,"metric":"cosine"}
```

768 is an example, not a dimension discovery result. Create a provider file:

```json
{"providers":["ollama"],"ollama":{"model_name":"your-installed-model","base_url":"http://127.0.0.1:11434","timeout":30}}
```

Explicitly create a **new, nonexistent** root; existing paths are never overwritten:

```sh
vecretary-init --data-dir /absolute/runtime --identity /absolute/identity.json
```

Initialization creates zero memories and a bound identity, not a provider health
check. A failed initialization may leave a partial root; inspect it and choose
a new path rather than retrying over existing files. Configure your MCP client to launch:

```sh
vecretary-mcp --database-dir /absolute/runtime/sqlite --data-dir /absolute/runtime --identity /absolute/runtime/embedding-identity.json --provider-config /absolute/local-provider.json --dashboard-port 8767
```

The same process owns the dashboard at `http://127.0.0.1:8767`; stopping MCP closes
both. Omit `--dashboard-port` for stdio-only operation. Launch logs go to stderr.
The dashboard has no user login or HTTPS: accepted local callers can read memory
data and obtain its write capability. The supported model is single-owner local
use, not shared-computer isolation or remote hosting. See the
[authentication/HTTPS assessment](docs/DASHBOARD_AUTH_SECURITY.md).
See [client example](examples/mcp.json.example) and
[agent instructions](examples/client-instructions.md). Without a provider binding,
metadata reads work but text search/content writes fail explicitly. Never open
untrusted databases or tools from a different storage profile against a SQLite runtime.

## Performance and release limitations

Exact scans remain linear. Synthetic 768-dimensional repeated scans on one Linux
host measured median 10.6 ms/230 records, 158.6 ms/1,000, 1.30 s/5,000 and
2.80 s/10,000. Cache limits are 512 records and 32 MiB accounted retained bytes,
not total process memory. These are sequential record-layer measurements,
excluding embeddings, transport, timer/pop dispatch and concurrency; not a
large-store interactive performance promise. No new vector index is included.

`runtime_ready=false` remains deliberate. A passing synthetic package gate is
not production certification or authorization to migrate a live installation.
An installed real-local-model check has also exercised text save/search through
stdio, owner dashboard boundaries and timer/access/keyword-pop persistence across
restart on isolated synthetic notes. That is not observed agent/chat/reload
acceptance, crash recovery or a provider performance guarantee.
The offline records-plus-sidecars migration scaffold is experimental; queue
disposition, real-copy acceptance, broader provider compatibility and crash
consistency remain separate gates. Legacy storage, lock, backup, and
shared-writer modules are not included in this
SQLite profile. The neutral offline migration exporter remains available for an
already-open collection from a separate trusted source environment.

## Validation scope

Prepublication RC26 hosted checks on Python 3.11.17/3.12.14 each passed 859 tests
with four optional-dependency skips. Complete combined runtime/test/scanner
audits covered 60 external distributions per profile with zero findings.
The actual loaded Ubuntu Noble SQLite library matched the separately reviewed
libsqlite3-0 3.45.1-1ubuntu2.8 package and bytes; this does not clear arbitrary
host SQLite builds. The new public repository has independent workflow runs;
check its Actions results for the exact public revision before relying on them.
Tests and advisory scans are not complete security or legal certification.

## Future importance feedback

A proposed next step is to use presence/frequency in vector recall as feedback
for retained importance, allowing repeatedly useful memories to stay prominent.
That new policy and semantic averaging are not added by this release. Existing
access/pop behavior and promotion rules remain unchanged.

## License, Wisdom and support

The public core uses plain [MPL-2.0](LICENSE), without premium-use restrictions.
Wisdom is a separate private/premium API module, not bundled or required by this
runtime. General support is best-effort via support@raw-wire.com; no response
time or service-level agreement is guaranteed. Premium support is offered separately.
See [Wisdom boundary](docs/WISDOM_BOUNDARY.md) and
[contribution policy](CONTRIBUTING.md): issues are welcome, external code
contributions are not accepted initially.

Report security issues privately to security@raw-wire.com; do not send real
memories, credentials or databases. See [security](SECURITY.md) and
[third-party notices](THIRD_PARTY_NOTICES.md). No vulnerability-free claim is made.

## Support development

[Sponsor Vecretary's developer](https://github.com/sponsors/roller11) to help fund
open-source maintenance, tests and documentation. Sponsorship is optional; it does
not change MPL-2.0 rights, include Wisdom, purchase premium support or guarantee
features or response times. General support is best-effort via
support@raw-wire.com, with no response-time guarantee or service-level
agreement. Sensitive vulnerability reports go to security@raw-wire.com.
