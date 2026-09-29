# Tool calling through MCP or APIs

Generate a complete Harbor task whose solution uses the assigned tools to carry out a meaningful
workflow: files and documents, structured records, several local services, or authorized remote
accounts. Toolathlon illustrates such workflows; copying its tasks, installing all its tools or
using its evaluation service is not required.

Read the supplied tool catalog and its exact schemas. The task may use only server commands and
endpoints the host allows. Generated service code or configuration must use the documented
isolation and startup route; it cannot choose privileged adapters, host mounts, credentials or
arbitrary network destinations. Never embed service tokens in public assets or the instruction.

Keep all task-specific initialization, fixtures, reference solution and verifier in the Harbor
package. `solution/solve.sh` can call a supplied MCP client or replay an authored sequence of
tool calls. A tool trace is an optional solution asset, not the package format. The reference
solution must use the same task-facing API as the solver, not write protected service state or
call evaluator-only operations.

`tests/test.sh` should inspect the resulting service state or artifacts and check task-specific
constraints, including side effects that must not happen. Recorded tool calls are separate
evidence; a fixed call count or a "done" tool is not a verdict. A failed service connection is an
execution failure, not a passing empty-solution result.

Reset every mutable service, database, account namespace and relevant workspace. Restarting an
MCP process does not reset the state behind its tools. Prefer bounded, reproducible local state;
real external applications need explicit credentials, scoped permissions, isolation and a reset
procedure.

A pattern that keeps tool tasks honest: declare the tools in a catalog file inside the task and
expose each one as an HTTP call to your own loopback service, so a catalog entry can never name
another destination, a file or a command. Run the service under an account the solver cannot
write to, keep its mutable state where the host can measure it for reset, and have setup wait for
the service's own response rather than for an open socket. If the runtime supplies a tool bridge
or a trusted service launcher, use its documented interface and nothing else.
