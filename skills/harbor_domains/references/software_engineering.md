# Repository-level software engineering

Generate a complete Harbor task whose goal requires understanding and modifying a repository.
The scenario can target a bug, a feature, a regression, a refactor or a build failure. Keep the
assigned failure evidence in view; do not claim a SWE-bench result just because the task uses a
repository.

Pin the repository revision and every build and test dependency in the permitted environment
construction route. You may author a new repository, or vary task-specific source and fixtures
where the assignment allows. Keep public requirements and inputs in the environment; keep
reference patches, hidden tests and reference outputs in the private `solution/` and `tests/`
directories.

`solution/solve.sh` must apply the authored change to the initial checkout and exercise the
requested workflow. A patch is an asset in the package, not a different output format.
`tests/test.sh` should run tests for the target behaviour and for relevant regressions. The
empty solution must fail because the original code is wrong, not because a secret success
marker is missing.

Running the tests necessarily executes solver-modified code: isolate it in the task runtime,
protect the verifier entry point and the reward files, and do not import the task into the host.
Do not let a solver-controlled test runner, `conftest.py`, Python path or package installation
replace the private verdict logic. Plan wrong-patch and incomplete-patch challenges.

Reset the repository, build outputs and relevant service state before every trial. Use a pinned
image or build context for large source trees.
