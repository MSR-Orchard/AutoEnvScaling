# Desktop (computer-use) workflows

Desktop tasks run on an assigned, digest-pinned base image (or virtual machine) that already
carries the desktop stack. The runtime's entry point, input bridge and readiness probe belong to
the host: do not copy, patch or shadow them, do not build or fetch a disk image, and do not try
to install system packages from an unprivileged setup. The runtime's application catalog decides
which applications are available; do not infer it from an old reference task.

You author the instruction, the setup that puts the machine into its starting state (for
example `environment/setup.sh` plus files under `environment/public/`), the reference solution
and the evaluator. Read assets from the path the runtime documents, not a guessed one. Set
`HOME` and `DISPLAY` before launching a GUI application, and poll for its window instead of
sleeping a fixed time. Applications that build a user profile on first launch (LibreOffice, for
example) are much slower the first time, so allow a generous timeout, match the window by more
than one signal, and on failure print the window list and the application's own log so the next
attempt can repair a named cause. Do not create the runtime's readiness marker yourself; setup's
only job is to exit 0 once the environment is genuinely ready.

The reference solution drives the desktop through the supplied mouse and keyboard bridge. It can
be an adaptive script that checks screenshots or the accessibility tree between actions; a fixed
JSON trace is optional.

## Reset must be byte-for-byte reproducible

The host checks that a fresh instance reaches the same starting state, and desktops break this
in two ways:

- Documents generated during setup embed creation and modification times, so identical inputs
  give different bytes. Ship the document as a fixed asset and copy it in setup; never generate
  or convert it there. Prefer flat XML formats (`.fodt`, `.fods`) when the task allows.
- Applications leave lock files beside open documents (LibreOffice writes `.~lock.<name>#` with a
  pid, host and time). Delete stale locks before launching.

Keep whatever the application opens writable by the solver: a read-only document makes
LibreOffice open it read-only and silently discard every edit. Enforce "do not alter the source"
by keeping a private reference copy under `tests/` and comparing against it.

## Drive the interface as something that answers

Most desktop reference solutions that fail do so because a dialog was still open and the
following keystrokes went into it. Three rules prevent this:

- Position the cursor by searching, not by counting: Find (`ctrl+f`), type the identifying text,
  Enter, Escape. Counting arrow keys from `ctrl+home` assumes a starting position you have not
  checked.
- Save last, in the format the document already has. Saving to a different format raises a
  second confirmation dialog that swallows the next keystrokes.
- Confirm before typing: check with a screenshot or the accessibility tree that a dialog closed
  and the cursor is where you expect, rather than waiting a fixed number of seconds.

## Grade the saved state

The verifier reads the application's artifacts or state after the solver finishes, not a
success marker the solver wrote. Grade the effective state, not one serialization of it: a GUI
application may attach a format to a cell, a row, a column or a named style, and may collapse
repeated runs, so resolve inheritance before asserting. Never require a locale-dependent
rendering (the same currency format can show as `$` or `HK$`). A verifier that checks only the
leaf attribute will score the reference solution zero.

The host records action and screenshot evidence and hashes the reset paths before each agent
phase; that evidence is for review, not proof that a task with clicks has meaningful GUI content.
Plan a wrong-output challenge. The reference solution must score 1, a do-nothing agent 0, and the
starting state must not already satisfy the task.
