# SQLite alpha release gates

Passing automated tests alone does not authorize publication or live migration.
This profile must be assessed separately from earlier Chroma candidates.

## Automated preparation

1. Explicit new-root creation, correct identity and refusal to overwrite existing
   paths. No implicit migration, seed, provider fallback or model download.
2. Isolated wheel metadata, MPL-2.0 license inclusion and SQLite console commands;
   Chroma must not appear in the installed runtime dependency inventory.
3. Actual installed stdio initialization/tools, loopback dashboard health and
   explicit refusal of unbound content writes. Synthetic provider regressions
   must preserve active-time timer, mandatory lifecycle and both pop mechanisms.
4. Linux Python 3.11/3.12 regressions, including pending queues, guards,
   observer immutability, backup/restore and validated exact linear scans.
5. Complete current advisory scan of resolved runtime dependencies without
   suppression. The effective SQLite/Python runtime requires separate review.
   Capture source identity, compile options and native linkage for every tested
   interpreter. A standalone Python can embed a different SQLite from the OS;
   distro backport evidence must not be applied to an unrelated embedded build.
   Runtime evidence collection does not itself grant advisory clearance.
   On Linux, record the actually loaded SQLite library SHA256, owning binary/
   source package versions and package-file checksum match. Missing provenance
   or a checksum mismatch fails collection; successful collection still needs
   build-specific advisory review. The package database is not an independent
   archive-signature check.
6. Allowlisted public source export and artifact hashes; targeted privacy
   patterns are not a substitute for final manual public-boundary review.

## Separate approval gates

- Actual Ollama model/provider behavior on isolated synthetic data.
- Native client behavior across session/chat/reload using the installed candidate.
- Full sidecar/auxiliary-state migration and crash/queue recovery policy before
  any live cutover. Database/sidecar commits are not atomic across files.
- Dependency/model licensing inventory and source provenance review. The owner
  attested in-house development and personally held rights; that is not a legal
  certification or third-party licensing clearance.
- Explicit final publication approval. Do not claim no vulnerabilities, production
  readiness, Windows support, indexed scalability or premium Wisdom availability.

Security reporting is security@raw-wire.com. The owner reports successful delivery
and forwarding; no alpha response SLA is promised. Issues are welcome; external
code contributions are not initially accepted. Premium support/Wisdom remain
separate from the MPL-2.0 core.
