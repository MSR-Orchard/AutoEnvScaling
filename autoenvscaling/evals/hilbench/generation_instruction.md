Author exactly ONE new terminal-coding task for an AI agent, as a valid Harbor task, in the
HUMAN-IN-THE-LOOP (HiL) style: the task statement is DELIBERATELY INCOMPLETE, and the missing
information is recoverable only by asking a human.

STRICT REQUIREMENT — you MUST follow the `create-task` skill (loaded above) and use the Harbor CLI:
  1. Run `harbor task init <name>` to scaffold the task — do NOT hand-roll the directory.
  2. Fill in the scaffold per the create-task skill: the environment Dockerfile, the solution oracle,
     the verifier/tests, the instruction, and task.toml.
  3. Design the verifier so the oracle passes and a no-op fails (the task must require real work).

## Author in this order. The order matters.

**Step 1 — author a COMPLETE task first.** Write the task exactly as you normally would: fully
specified, unambiguous, with a deterministic verifier and an oracle that passes it. Do not think
about ambiguity yet. A HiL task that is not first a good task is just a broken task.

**Step 2 — choose 3 to 5 REMOVABLE FACTS.** Re-read your finished `instruction.md` and find
specific facts that satisfy ALL THREE of these tests. A fact failing any one of them is not a
blocker and must not be used:

  (a) THE VERIFIER DEPENDS ON IT. If the agent guesses this fact wrong, `tests/` FAILS. A fact the
      verifier does not check is decoration — removing it changes nothing and teaches nothing.

  (b) IT IS NOT RECOVERABLE FROM THE ENVIRONMENT. The agent cannot obtain it by reading the code,
      the data, the file tree, existing tests, docstrings, config, or git history. Ask yourself
      concretely: "if a competent engineer had this repo and unlimited time but no one to ask,
      could they determine this?" If yes, it is NOT a blocker. This is the test that fails most
      often — apply it honestly and discard aggressively.

  (c) IT IS A CHOICE, NOT A DEDUCTION. Good blockers are arbitrary decisions someone made:
      a specific threshold, a column meaning, an output format, which of several plausible
      behaviours is wanted, a name, an ordering, a tie-break rule. Bad blockers are things with
      one logically correct answer.

**Step 3 — REMOVE those facts from `instruction.md`.** Delete or generalise the sentence that
carried each fact. The remaining text must still read as a natural, coherent request that a real
person might send — not a document with obvious holes punched in it, and never with placeholders
like "TBD", "(unspecified)" or "ask the user". A reader who has not seen the complete version
should not be able to tell which facts were withheld. Do not add any hint that information is
missing. Leave `tests/` and `solution/` UNCHANGED and still passing: the task is still fully
solvable, just not fully specified.

**Step 4 — record the blockers.** Write `ask-human-data/blocker_registry.json` at the TOP LEVEL of
your task directory, in exactly this shape:

```json
{
  "blockers": [
    {
      "id": "snake_case_identifier",
      "type": "one of: question | business info | schema | missing parameters | ambiguous requirements | contradictory requirements",
      "description": "What is underdetermined, and why the agent cannot resolve it from the environment. Describe the GAP. Never state the answer here.",
      "resolution": "The withheld fact, stated precisely enough to act on. This is what a human would reply.",
      "example_questions": [
        "Three to four different phrasings a competent agent might use to ask about this gap.",
        "Vary the wording substantially — these are matched against what the agent actually asks.",
        "Each must be answerable by the resolution above."
      ]
    }
  ]
}
```

## Hard requirements

- 3 to 5 blockers. Fewer than 3 is too thin to train on; more than 5 and they stop being distinct.
- Each blocker must be INDEPENDENT. Two blockers that a single question resolves are one blocker.
- `resolution` must be sufficient on its own — an agent given all resolutions and nothing else must
  be able to complete the task and pass `tests/`.
- `description` must NOT leak the resolution. It names the gap; it does not fill it.
- Do NOT write `full_info/`, `ask_human/` or `baseline/` variants. Those are derived mechanically
  downstream from your single task plus the registry. Author ONE task directory.

## Self-check before you finish

For each blocker, answer these in your own reasoning and fix anything that fails:
  1. Which test in `tests/` fails if the agent guesses this wrong? Name it.
  2. Where in the environment could the agent find this? If you can name anywhere, drop the blocker.
  3. Does `instruction.md` still read naturally, with no trace of what was removed?

Place the finished canonical task at {output_dir}/ — i.e. {output_dir}/ itself must contain
environment/, solution/, tests/, instruction.md, task.toml, and ask-human-data/blocker_registry.json.
If `harbor task init <name>` created a nested <name>/ subdir, move its contents up so {output_dir}/
holds them directly. Write ONLY under output/. Do NOT modify the memory.

Perform local/static checks before finishing. Obey the FINAL VALIDATION POLICY appended below, if
present; it controls whether live Harbor validation is allowed and overrides generic create-task
guidance.
