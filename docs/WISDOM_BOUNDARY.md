# Vecretary Memory and Wisdom product boundary

Status: owner-approved product/license boundary; no new plugin API implemented.

- Vecretary Memory remains the standalone memory MCP server and local dashboard.
- Wisdom is a planned, separately released optional premium module with its own
  package, repository, release cadence and dependencies. Its source availability,
  license and distribution terms remain to be specified by the owner.
- Do not bundle premium implementation, model credentials, paid-provider
  dependencies or Wisdom configuration into Vecretary Memory's default distribution.
- Prefer existing MCP contracts or a reviewed, documented extension interface.
  Vecretary Memory must operate and pass its default tests without Wisdom installed.
- Keep shared contracts small and explicit. Integration does not grant Wisdom
  elevated memory-write authority or bypass owner-only operations.
- Existing Wisdom-named compatibility aliases in Vecretary Memory do not constitute
  the premium Wisdom product. New plugin hooks remain a separate design task.

## Approved licensing direction

The owner selected plain MPL-2.0 for Vecretary Memory. Wisdom remains outside the
core distribution and API-separated, with its own licensing and distribution
decisions. A separate premium release is planned; this does not select Wisdom's
license or authorize bundling its implementation into Memory. Vecretary Memory
accepts issue reports but no external code contributions during the initial
release period. This policy does not restrict rights granted by MPL-2.0.

Premium support is planned as a separate offering, not a license condition.
Neither Wisdom nor a support purchase is required to install, use, modify or
redistribute Memory subject to MPL-2.0. No pricing, availability date, support
scope or service-level commitment is established by this document.

Memory's canonical LICENSE and package metadata remain MPL-2.0. Charging for
support or offering an optional premium module does not relicense the core.

Mozilla's [MPL FAQ](https://www.mozilla.org/en-US/MPL/2.0/FAQ/) explains that
MPL-covered code may be combined with separately licensed proprietary code.
Distributing MPL-covered executables still requires making the covered source
available as required by the license. Keeping Wisdom in another repository
does not exempt copied or modified MPL-covered files from their obligations.

Recommended engineering separation: keep premium implementation in new files
in Wisdom, use public contracts, and review any modifications to Vecretary Memory
separately. Repository separation alone is not legal clearance or assurance of
ownership of product names. Dependencies retain their own licenses; MPL-2.0
does not relicense Chroma or other third-party code.
