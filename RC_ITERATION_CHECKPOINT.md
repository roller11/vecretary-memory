# RC Iteration Checkpoint

Updated: 2026-10-12

This is an engineering checkpoint, not release acceptance or a claim that every
listed change has been pushed. The working tree is based on
`sqlite-alpha-0.2.0a2-staging` at `d978bed` and contains uncommitted iteration
changes. Do not publish until the candidate is reviewed and validated.

## Data boundary

- No live memory records were migrated, copied, or used as RC fixtures.
- RC initialization and tests must use only the sanitized starter profile and
  disposable synthetic data.
- No live service restart, configuration switch, or database operation is part
  of this source checkpoint.

## Changes recorded in the working tree

- Dashboard and SQLite service: explicit MCP/dashboard write-control handoff,
  visible mode/status controls, heartbeat lease handling, safe refusal when a
  competing writer owns the guard, and capability/origin checks. Handoff is a
  logical transfer within the service; it does not terminate the host-owned
  MCP process. Queue-capture mode does not support handoff.
- Mandatory reminders: renew active mandatory context to score 99 on session
  restoration and at the configured turn interval (five turns by default);
  after model access, the reminder elevation returns to the 79 baseline.
- Runtime and retrieval: SQLite session/turn state, active-time handling,
  mandatory lifecycle and scope behavior, and service/API integration updates.
- Dashboard styling: the final refined live UI was NOT merged into this RC
  before the handoff r1 wheel was built and installed. That installation
  overwrote the refined UI, including the overview. The subsequent standalone
  navigation/theme changes do not recover the complete redesign. This is a
  package overwrite, not a cache problem. Merge the preserved final UI patch
  with the newer handoff/pop changes before building another candidate.
- Documentation and regression coverage: README/release notes, SQLite service
  integration documentation, and tests covering session restoration, access
  decay, turn intervals, service handoff, and SQLite service boundaries.

## Changed files in the current tracked diff

- `README.md`, `RELEASE_NOTES.md`
- `docs/SQLITE_SERVICE_INTEGRATION.md`, `docs/sqlite-release/README.md`
- `instinct/dashboard.py`, `instinct/mcp_server.py`, `instinct/mcp_tools.py`
- `instinct/models.py`, `instinct/nudge.py`, `instinct/retriever.py`
- `instinct/sqlite_runtime.py`, `instinct/sqlite_service.py`,
  `instinct/sqlite_session.py`
- `tests/test_active_time.py`, `tests/test_dashboard_features.py`
- `tests/test_mandatory_lifecycle.py`, `tests/test_mandatory_scope.py`
- `tests/test_nudge_session.py`, `tests/test_seed.py`
- `tests/test_sqlite_runtime.py`, `tests/test_sqlite_service.py`,
  `tests/test_sqlite_session.py`

## Validation state

- The prior full RC run reported 1,127 passed and 22 skipped before the latest
  standalone-page CSS update; do not treat that as validation of this final
  checkpoint.
- The focused service test could not be discovered by the test tool because
  the RC repository is outside the active VS Code workspace.
- Editor diagnostics reported no errors and `git diff --check` passed after
  the latest CSS change.
- Re-run the focused service tests, then the full configured RC suite, before
  packaging or pushing this candidate.

## Verified UI regression provenance - 2026-10-12

- The previous refined dashboard matches the final UI patch manifest at SHA-256
  `06133575aadf904e556fe278c8bd4fb47736ac0948771382ce3c3fd50b5ee6cb`.
  Its source and pre-install package backup remain preserved.
- The handoff r1 wheel and currently installed dashboard match at SHA-256
  `f3711ab76e8aab7524089f639bfcfd8fd78896860cfd791afa56891c8b153aae`.
  This confirms a package overwrite, not a browser-cache problem.
- A fresh uncached response on 8766 lacks the refined dark accent `#176331`,
  light background `#d9dcda`, secondary app bar, and 256px logo reference.
- The complete exported UI patch has SHA-256
  `da499a33f5c1a4058449c2dccb20b2f151892cef25e47252179ab67484adf16e`.
  Merge its UI and asset-route changes; do not replace newer handoff backend
  files wholesale with the older UI-only source.
- This investigation did not change live code, restart services, or access
  memory records. No commit or push was performed.

## Integrated recovery - 2026-10-12

- Merged the preserved complete UI patch with the newer handoff/pop work;
  restored the packaged 256px logo and its exact SQLite GET allowlist route.
- Removed the conflicting secondary-header implementation and used the refined
  shared header/theme instead. Added shared link boundaries and spacing so
  secondary navigation does not collapse into unformatted text.
- Regression coverage now checks overview, decay, score, and modifier routes.
  Updated the browser capability test DOM fixture for the shared header.
- Native Windows source suite: 1,133 passed, 22 skipped before adding modifier
  route coverage. Final targeted dashboard/service/security run: 169 passed.
- Linux source suite with Node: 1,152 passed, four skipped. Installed isolated
  smoke and dependency consistency pass; dependency advisory audit reports PASS
  with no findings. These are bounded tests, not release security certification.
- Added full Linux regressions and experimental native Windows installed,
  functional/security, and advisory checks to hosted CI. See docs/RC_TESTING.md.
- Separate test installations use the same candidate wheel and only three
  bundled sanitized starter memories. No live database is an RC fixture.
