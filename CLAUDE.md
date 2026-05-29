# CLAUDE.md

Start with [`README.md`](README.md) — it covers project architecture,
build, install, dev workflow, testing, and style. This file lists
operational rules that matter when an AI assistant is editing the
codebase.

## Commit messages

Follow Chris Beams' "How to Write a Git Commit Message"
(<https://cbea.ms/git-commit/>): imperative subject ≤50 chars, blank
line, body wrapped at 72 explaining *what* and *why*.

Do not reference internal scaffolding from commit messages. Don't
mention `plan.md`, phase numbers, or AI-internal organization. Commits
describe the code change on its own terms.

## Docstrings and comments

Docstrings and comments describe what the code does **today**, in
declarative present tense. Do not narrate hypothetical bad
alternatives, prior implementations, or change history. No
"previously," "used to," "would otherwise," "imagine if we did X
instead." History belongs in commit messages and PR descriptions.

If you're tempted to explain *why* the code looks the way it does
in terms of an alternative, stop and rewrite to name the actual
current constraint or invariant. Future-you reading this in a year
needs to know the rule, not the road not taken.

## Working files

- **`CLAUDE.local.md`** — local sandbox/dev notes and opinions.
  Gitignored. Update as work progresses.

## Naming conventions

- New CMake options use the `PHOENIX_*` prefix to match the existing
  build (e.g. `PHOENIX_ENABLE_STRICT_DEV_ENV`).
- The `dev` CMake preset and editable pip installs enable
  `PHOENIX_ENABLE_STRICT_DEV_ENV=ON` by default.
