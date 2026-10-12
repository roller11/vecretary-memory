# Third-party licensing - SQLite profile

Vecretary project source uses plain [MPL-2.0](LICENSE). Dependencies retain their
own licenses; the project license does not replace them. Python/SQLite and
the installed MCP/HTTP dependency closure require separate inventory review.
This document is not a complete native-library provenance inventory or legal
opinion. The installed-artifact evidence collector records actual license texts
and hashes; missing license files fail its check. Collection is not automatic
legal clearance.

## Observed Python package inventory

The isolated Linux Python 3.11/3.12 candidate environments were inspected on
October 7, 2026. Both contained the same 31 external package versions below,
plus the MPL-2.0 Vecretary core. Every installed distribution supplied license
evidence. These are declared package licenses, not an assertion that each
native or vendored component has only that license.

| Package | Version | Declared license |
| --- | --- | --- |
| annotated-types | 0.8.0 | MIT |
| anyio | 4.15.1 | MIT |
| attrs | 26.1.0 | MIT |
| certifi | 2026.7.22 | MPL-2.0 |
| cffi | 2.1.1 | MIT-0 |
| click | 8.5.0 | BSD-3-Clause |
| cryptography | 50.0.2 | Apache-2.0 OR BSD-3-Clause |
| h11 | 0.16.0 | MIT |
| httpcore | 1.0.9 | BSD-3-Clause |
| httpx | 0.28.1 | BSD-3-Clause |
| httpx-sse | 0.4.3 | MIT |
| idna | 3.20 | BSD-3-Clause |
| jsonschema | 4.26.0 | MIT |
| jsonschema-specifications | 2025.9.1 | MIT |
| mcp | 1.30.0 | MIT |
| pip | 26.2.1 | MIT |
| pycparser | 3.0 | BSD-3-Clause |
| pydantic | 2.13.5 | MIT |
| pydantic-settings | 2.15.0 | MIT |
| pydantic_core | 2.46.5 | MIT |
| PyJWT | 2.15.1 | MIT |
| python-dotenv | 1.2.4 | BSD-3-Clause |
| python-multipart | 0.0.32 | Apache-2.0 |
| referencing | 0.37.0 | MIT |
| rpds-py | 2026.9.1 | MIT |
| setuptools | 83.0.0 | MIT |
| sse-starlette | 3.5.0 | BSD-3-Clause |
| starlette | 1.7.0 | BSD-3-Clause |
| typing-inspection | 0.4.4 | MIT |
| typing_extensions | 4.16.0 | PSF-2.0 |
| uvicorn | 0.54.0 | BSD-3-Clause |

Pip and setuptools are installed packaging/bootstrap tools, not direct core
runtime imports. Their vendored notices are collected separately within their
license evidence. Dependencies are resolved separately at installation: this
source draft and project wheel do not bundle their distributions. Preserve
their supplied license/copyright/notice files when redistributing them. Review
native bundled-library obligations before shipping an all-in-one installer.

## Python, SQLite and optional model

Python and the operating-system SQLite library are outside the Python package
inventory. Keep the notices supplied by their distributors. Their patch status
and native build provenance need separate review; a clean Python package scan
does not establish system-library security clearance.

The tested local nomic-embed-text:latest model reports an Apache-2.0 license in
its installed provider metadata. Neither that model nor Ollama is bundled or
downloaded by the Vecretary initializer. Users must separately review the
chosen provider/model's terms and provenance; provider-reported metadata is
not independent model/training-data licensing certification.

No legacy vector-store library, native wheel or server is bundled by this profile.
The offline migration helper only accepts a caller-supplied open collection and
does not bundle or construct its source database client.

Wisdom implementation, credentials and premium dependencies are not included.
See [API boundary](docs/WISDOM_BOUNDARY.md). Premium support is separate.
