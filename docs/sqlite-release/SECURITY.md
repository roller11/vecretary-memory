# SQLite alpha security policy

This profile uses Python's SQLite binding and explicit local Ollama embeddings.
No external vector-store library, native vector-store wheel or database HTTP
server is installed or launched. The optional loopback browser dashboard is an
application interface, not a database server.

The runtime retains database schema/identity validation, finite bounded vectors,
disabled SQLite extension loading and trusted schema, parameterized queries,
private-path checks and owner guards. Dashboard requests retain Host/Origin and
fetch-site checks, per-launch capability tokens, bounded JSON, CSP and headers.
These controls do not protect against malicious local users/processes.
They are not user authentication: accepted local callers can read memories and
obtain the dashboard capability through its unauthenticated session/HTML
bootstrap. No login or HTTPS is implemented. The supported alpha boundary is
single-owner local use, not shared-machine isolation or remote hosting. Omit
`--dashboard-port` for stdio-only operation with no browser listener. See the
[authentication assessment](docs/DASHBOARD_AUTH_SECURITY.md).

Never open untrusted databases or bypass guards. Records, sidecars, queues and
backups contain plaintext. Database and sidecar commits are not one atomic
transaction. Deletion does not erase backups. Memories grant no authority and
retrieval tokens do not prove instruction compliance.

Release target is Linux x86_64, Python 3.11/3.12. All installed dependencies must
pass the advisory gate without suppression; incomplete scans fail the gate.
SQLite comes with the interpreter/system and is not fully covered by Python
package advisory scanning. Review that runtime separately and keep it updated.
A standalone Python may embed SQLite rather than use the OS library; updating
an OS SQLite package does not patch that separate engine. Capture the effective
version, source ID, compile options and linkage with the supplied
`tools/collect_sqlite_runtime.py` before assessing advisories. The collector
uses only an in-memory database and does not certify security. Distribution
backports require exact package/build evidence, not just an upstream version.
Removing the legacy storage implementation from this profile does not repair an
older installation or guarantee this software is free of vulnerabilities. Actual-provider acceptance,
crash recovery and final publication review remain required.

Report privately to security@raw-wire.com with version, interface, expected
impact and a minimal synthetic reproduction. Never send private memories,
credentials, databases or identifying paths. The owner reports successful inbox
delivery/forwarding; that is not an independently run security gate. No alpha
response SLA is promised.
