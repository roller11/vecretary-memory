# SQLite isolated client acceptance

Use a fresh synthetic SQLite root and the reviewed installed SQLite commands,
not legacy seed/client-generation helpers or a real memory store. Do not
change global client settings, remove guards or restart a live deployment.
Automated transport tests and observed agent behavior are separate gates.

1. Create a new empty root with `vecretary-init` and an explicit identity matching
   an already-installed local Ollama model. Launch the SQLite MCP command with
   that root, identity and matching provider configuration.
2. Configure only a separately reviewed isolated client workspace. Confirm the
   listed delete tool moves records to the recoverable recycle bin, not hard delete.
   Fetch context/mandatory records, confirming
   empty-store behavior and truthful reports of unavailable functionality.
3. Explicitly authorize one synthetic ordinary score-70 save, fetch its returned
   ID and verify its exact content. Exercise search and score update; confirm
   unauthorized mandatory/high-score writes are refused.
4. Inspect the owner's loopback dashboard and its capability-protected mutations.
   Confirm active-time ticks, access/pop diagnostics and both configurable decay
   periods without changing score formulas or using real records.
5. Repeat after stopping/restarting only that isolated process and after a new
   chat/window reload. Confirm persistence, fresh mandatory lifecycle behavior
   and absence of timer charging for idle wall time.
6. Review observer and pending-write behavior separately. Queued capture is not
   a committed save; a cancelled call does not undo accepted owner work. Check
   persistence before retrying. Permanent deletion remains unsupported; do not use
   the legacy acceptance protocol's delete/reset steps.

This protocol is not a private-data migration or production clearance. Keep
synthetic acceptance artifacts isolated; do not publish identifying runtime paths.
