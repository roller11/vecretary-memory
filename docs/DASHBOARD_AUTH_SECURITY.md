# Dashboard authentication and HTTPS assessment

Status: design recommendation and verified current boundary, not a new login or
TLS implementation. The supported alpha model remains single-owner local stdio
MCP with an optional loopback dashboard. Shared-computer isolation and remote
dashboard hosting are not supported by the existing controls.

## What is implemented now

The SQLite dashboard binds only `127.0.0.1`. Host, Origin and fetch-site checks
reject untrusted browser origins and unexpected destinations. Mutating requests
require a fresh per-launch capability token. Responses use security headers,
bounded encoding and a content security policy.

These are request-boundary and cross-site protections, **not user authentication**.
Any local caller that can reach the port and send accepted headers can read
memory API responses and request `/api/session`, which returns the mutation
capability. The HTML also bootstraps that capability. Local processes can send
accepted Host/Origin headers; those headers are not proof of identity. There is
no password prompt, authenticated session or TLS listener in this alpha.

An isolated real-provider check confirmed unauthenticated list/session reads,
refusal of mutations without a capability, acceptance with the locally obtained
capability and rejection of untrusted Host/Origin requests. This does not imply
that hostile websites can read the token: browser same-origin restrictions and
the explicit cross-origin boundary remain separate controls.

## Login versus HTTPS, in plain terms

- **Login controls who may access the dashboard.** A useful owner login must
  cover memory reads, exports, HTML/token bootstrap and writes, not just the edit
  buttons. A login page that leaves `/api/list`, `/api/get` or token bootstrap
  open would not meaningfully protect retained memories.
- **HTTPS protects data while it travels between browser and server.** Normal
  HTTPS also authenticates the server to the browser when certificates are
  validated. It does not identify the human user, fix database flaws, encrypt
  stored memories/backups or block software running as the same OS owner.
- **Both add responsibilities.** Authentication requires secure password/secret
  handling, session expiry/logout, request forgery protection and abuse controls.
  HTTPS requires trustworthy certificates and correct renewal/configuration.
  Skipping certificate validation or suppressing browser warnings defeats an
  important part of its value.

## Recommendation for this RC

Do not widen the listener or rush a custom account system into the SQLite alpha.
Retain the explicit single-owner local boundary and allow `vecretary-mcp` to run
without `--dashboard-port` when no browser access is needed. This removes the
dashboard listener entirely; stdio MCP operations remain available.

An optional **single-owner access gate** is a reasonable next hardening feature,
especially for shared computers or sensitive retained memories. It does not need
sign-up, user roles, password recovery, email or a multi-user database. Review the
implementation and its threat model separately before claiming stronger access
isolation. A per-launch secret is an alternative to maintaining a password file,
but it must not be obtainable from an unauthenticated HTTP endpoint.

If password sessions are selected, use reviewed password hashing and session
primitives, HttpOnly/SameSite cookies, expiry/logout, login throttling and the
existing CSRF/capability controls. Under TLS, session cookies must be Secure.
Protect all sensitive GET/POST routes and test read/token bootstrap refusals,
restarts, logout and brute-force behavior. Do not promise protection from malware
that already has the owner's OS/file access.

For any future remote mode, authentication **and** validated HTTPS are necessary,
but not sufficient. Prefer a reviewed TLS/authentication proxy rather than a
new bespoke network login stack; proxy trust, header handling, exposure limits,
rate limits and origin checks still require review. The current boundary accepts
only HTTP loopback origins: wrapping the existing socket with TLS or exposing it
through a proxy is not a supported configuration switch.

Authentication/TLS do not alter active-time decay, scores, pop rules or the MCP
agent's memory-write authority. Keep owner web authentication distinct from
agent authority. No live configuration or listener was changed by this assessment.

## Primary guidance

- [OWASP authentication guidance](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)
- [OWASP transport security guidance](https://cheatsheetseries.owasp.org/cheatsheets/Transport_Layer_Security_Cheat_Sheet.html)
- [OWASP session management guidance](https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html)

The authentication and TLS pages were retrieved directly on October 7, 2026.
This document is an engineering assessment, not an independent security audit.
