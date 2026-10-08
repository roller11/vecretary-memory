# SQLite alpha RC preparation

The SQLite profile uses a separate public-export package definition and README;
the legacy development defaults and live deployment remain unchanged.
`vecretary-init` explicitly creates an empty new runtime;
`vecretary-mcp` and `vecretary-sqlite` select the SQLite stdio/dashboard service.
There are no Chroma, standalone legacy dashboard, seed or backup entrypoints.

Exact linear scans, score rules, active-time timer and both pop pipelines remain.
The SQLite service retains `runtime_ready=false` and its documented unsupported
routes. This candidate is draft-only until installed-package, dependency,
provider, license/provenance and public-boundary gates are reviewed. Legacy
gate documents describe older candidates and do not certify this profile.

Synthetic tests do not authorize migration or release. Real-data conversion,
sidecar/crash recovery, live service changes and publication need separate review.
