# Integrated RC testing

The integrated candidate combines the refined dashboard, packaged 256px logo,
shared secondary-page navigation/theme, SQLite MCP/dashboard handoff with lease
recovery, and periodic mandatory reminders (99 on renewal, 79 after access).
Do not rebuild from an older source tree or replace the handoff backend with
the earlier UI-only package.

## Review and data boundary

Push the reviewed commits to the RC staging branch and open an owner PR to
`main`. Hosted checks and owner review precede merge; a branch push is not a
published release or production/security certification. This internal owner
process does not change the policy against external code contributions.

Use only fresh disposable data and the bundled sanitized starter memories.
Never package a live database, backup, logs, client configuration, or private
deployment helpers. UI and handoff tests must not modify real memories.

## Isolated installed checks

Use the same reviewed wheel for both platforms. The dependency constraint file
currently records the reviewed Linux versions; it can also constrain the
experimental Windows environment, but is not a Windows security certification.

Linux, from a reviewed source checkout:

```sh
python3.12 -m venv .venv-rc-linux
.venv-rc-linux/bin/python -m pip install --upgrade pip==26.2.1 setuptools==83.0.0
.venv-rc-linux/bin/python -m pip install -c requirements-linux.txt /path/to/reviewed.whl pytest pytest-asyncio pip-audit
.venv-rc-linux/bin/python -m pip check
.venv-rc-linux/bin/python -I tools/check_sqlite_installed.py
.venv-rc-linux/bin/python -m pytest -q
.venv-rc-linux/bin/python tools/audit_dependencies.py --output build/rc-linux-audit.json
```

Native Windows PowerShell, from a reviewed source checkout:

```powershell
py -3.12 -m venv .venv-rc-windows
& .venv-rc-windows/Scripts/python.exe -m pip install --upgrade pip==26.2.1 setuptools==83.0.0
& .venv-rc-windows/Scripts/python.exe -m pip install -c requirements-linux.txt C:/path/to/reviewed.whl pytest pytest-asyncio pip-audit
& .venv-rc-windows/Scripts/python.exe -m pip check
& .venv-rc-windows/Scripts/python.exe -I tools/check_sqlite_installed.py
& .venv-rc-windows/Scripts/python.exe -m pytest -q
& .venv-rc-windows/Scripts/python.exe tools/audit_dependencies.py --output build/rc-windows-audit.json
```

Install Node before running the browser capability tests; otherwise those tests
skip. Installed checks use isolated Python and synthetic temporary roots. The
full suite exercises checkout code; these are complementary, not interchangeable
checks. Review skipped tests rather than treating a pass count as full coverage.

## Browser acceptance

On the installed candidate, check `/`, `/decay`, `/score`, and `/modifier`:

- The 256px packaged logo loads from the same origin.
- There is one navigation group and one theme control, with visibly separated
  links, padding and boundaries on secondary pages.
- Dark accent is `#176331`; light background is `#d9dcda` and persists between
  routes and reloads. Check desktop, tablet and narrow layouts.
- Take Dashboard Control pauses MCP tools and enables authorized dashboard
  mutations. Return to MCP resumes tool calls. The browser lease returns
  control after heartbeat expiry; it does not launch a new host-owned process.
- Missing capability, hostile Host/Origin, unsupported routes, competing
  writers, and capture-mode handoff fail closed.
- Reminder interval/access tests confirm renewal to 99 and return to 79;
  the service counts observed MCP lifecycle turns, not unobserved chat turns.

Native Windows remains experimental. POSIX permission/ownership tests do not
certify Windows ACLs. WSL testing is Linux testing, not a clean native Windows
installer/VM certification. Hosted Windows checks are an additional regression
signal, not a change to the released support policy.
