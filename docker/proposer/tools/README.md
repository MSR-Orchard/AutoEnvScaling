# Proposer tools

These commands are on the proposer's PATH inside the sandbox. The image installs them in
`/opt/aes/tools`; on the first round they are also copied into the workspace's `tools/`
directory, which comes first on PATH, persists across rounds, and is the proposer's to edit
(the paper's harness optimization). The host's admission checks do not use these files, so
editing them cannot change what is admitted.

| Command | What it does |
|---|---|
| `aes-validate <task>` | Builds the task in child containers and runs Harbor's oracle agent (the reference solution must score 1.0) and nop agent (an empty solution must score below 0.5). |
| `solver_model <task> [-n N]` | Runs the current solver N times through `autoenvscaling.calibration.solver.run_solver` and reports rewards, spread and the calibration verdict. |
| `web_search "<query>"` | Web search through Serper (`SERPER_API_KEY`), with blocked benchmark sites removed. |
| `web_fetch <url>` | Fetches a public page as text; refuses blocked hosts and private addresses. |

Environment variables the host forwards into the sandbox:

| Variable | Used by |
|---|---|
| `AES_SANDBOX_HARBOR_ENV` | Harbor environment type for child containers (default `modal`) |
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | child containers on Modal |
| `AES_VALIDATION_REFERENCE_MIN_REWARD`, `AES_VALIDATION_EMPTY_MAX_REWARD` | `aes-validate` thresholds |
| `AES_SOLVER_BASE_URL`, `AES_SOLVER_MODEL`, `AES_SOLVER_API_KEY` | `solver_model` |
| `AES_CALIBRATION_*` | `solver_model` band and default attempt count |
| `SERPER_API_KEY` | `web_search` |
| `AES_BLOCKED_HOSTS`, `AES_BLOCKED_URL_PATTERNS` | `web_search`, `web_fetch` |
