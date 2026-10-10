# Vecretary Memory 0.2.0a3 - SQLite alpha

## Release update

This follow-up updates the package version and downloadable artifacts from
0.2.0a2. It aligns release-policy/runtime tests and default pytest discovery
with the supported SQLite alpha profile. It contains no application behavior
changes; supported features and limits remain the same.

## Included

- Local embedded SQLite storage with exact linear vector recall; Chroma is not
  an installed runtime dependency and no database HTTP server is started.
- Agent/vendor-agnostic stdio MCP context, mandatory retrieval, semantic search,
  individual reads, ordinary saves and score updates.
- Importance/modifier scoring, active-session timer, mandatory renewal, decay
  configuration, access tracking and both existing pop mechanisms.
- Local owner dashboard, read-only observers, opt-in pending capture/replay,
  offline backup/restore, queue review and confirmed recycled-record purge.
- Plain MPL-2.0 core; optional premium Wisdom and support remain separate.
- Optional [GitHub sponsorship](https://github.com/sponsors/roller11), with no
  purchase requirement, premium entitlement or response-time guarantee.

## Supported scope and limits

Linux/WSL x86_64, Python 3.11/3.12, local single-owner use. Native Windows/macOS,
remote hosting, multi-user isolation and login/HTTPS are not supported by this
alpha. Use a new isolated runtime and an explicitly bound, already-installed
Ollama embedding model. No automatic model download or legacy-store migration.

Exact scans are linear, not indexed. Database and lifecycle/configuration/queue
sidecars are not committed atomically across files. Queued capture is not a
committed save. Agent hard deletion, online backup/reset and writer handoff
remain unsupported. Keep private runtime/queue/backup files out of the repository.

This release includes read-only native SQLite provenance collection and pins
the CI runner distribution to Ubuntu 24.04. It records loaded-library hashes and
package checksum evidence; collection alone is not advisory clearance. The
prepublication RC26 passed hosted Python 3.11/3.12 checks, each with 859 passed
and four optional skips, and 60-distribution audits with zero findings. Hosted
SQLite bytes matched the reviewed Ubuntu package. Public repository workflows
verify their own revision; no result certifies arbitrary host SQLite builds.
The tagged alpha provides a wheel, exact-revision source ZIP, installation
constraints and SHA256 checksums. It is a prerelease, not a stable release.

No artwork is bundled in this candidate. Core scoring, timers and pop behavior
are unchanged. A proposed vector-recall importance-feedback policy is not
implemented. Neither production readiness nor vulnerability-free status is claimed.

## Contact

General support: support@raw-wire.com. Sensitive security reports:
security@raw-wire.com. Issues are welcome; external code contributions are not
accepted initially. Premium support is separate; no alpha SLA is promised.
