# Vecretary Memory 0.2.0a2 - SQLite alpha candidate

## Included

- Local embedded SQLite records with exact linear vector recall; no Chroma
  runtime dependency or database HTTP server.
- Agent-agnostic stdio MCP retrieval, saving, score updates, and recoverable
  recycle, with the active-session clock, mandatory lifecycle, decay, and both
  existing pop behaviors retained.
- Active mandatory scores 99-100 are periodically resurfaced at 99 (five turns
  by default); accessing them lowers their score to 79 until the next nudge.
- Local dashboard, observer mode, opt-in pending-write capture/replay, offline
  backup/restore, queue review, and explicitly confirmed recycled-record purge.
- Refined charcoal/green and neutral-light dashboard with shared secondary-page
  navigation, saved themes, and the packaged 256px logo.
- Capability-protected MCP/dashboard write-control handoff with browser lease
  recovery; the host-owned stdio process remains attached.
- Namespaced additive score extensions for future modules.
- Opt-in, offline migration tooling that preserves record metadata, embeddings,
  recognized runtime sidecars, and pending queue bytes. Migration does not replay
  queues or make the imported runtime ready.

## Candidate verification

The October 12 integrated iteration restores the complete UI without reverting
the handoff/reminder work and adds experimental native Windows hosted checks.
See [integrated RC testing](docs/RC_TESTING.md). The older verification below
records a prior milestone, not the final integrated candidate's test result.

The controlled Windows Python 3.12.5 run passed 1,108 tests with 18
platform-specific skips. A fresh local wheel install passed dependency
consistency, console entry point, loopback dashboard, stdio MCP, and unbound
write-refusal checks. This is not hosted Linux CI or native SQLite advisory
clearance.

Actual-store acceptance is incomplete. No live Chroma store was opened or
migrated. Do not use this candidate for live migration or treat synthetic
migration/rollback tests as evidence about existing user data. Linux Python
3.11/3.12 hosted CI, a stopped byte-verified disposable real-data migration,
queue disposition review, and separate cutover approval remain required.

## Scope

This is a prerelease candidate, not stable or production-certified software.
Supported target remains Linux/WSL x86_64, Python 3.11/3.12, local single-owner
use. Native Windows/macOS, remote hosting, multi-user isolation, login/HTTPS,
and indexed vector search are not supported. Keep private databases, memory,
queues, backups, and operational configuration out of source archives.

The SQLite migration importer keeps \`runtime_ready\` false. Staged-import rollback
does not provide automatic live-service rollback or prove power-loss behavior
on every filesystem. No vulnerability-free or legal-certification claim is
made. The vector-recall-dependent importance proposal is not implemented.

## Contact

General support is best-effort via support@raw-wire.com; no response time or
service-level agreement is guaranteed. Sensitive security reports:
security@raw-wire.com. External code contributions are not accepted initially.
Premium Wisdom and support remain separate from the MPL-2.0 core.
