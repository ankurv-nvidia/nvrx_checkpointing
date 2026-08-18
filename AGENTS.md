# Repository Instructions

## Instruction Precedence

- Direct user instructions take precedence over all repository guidance.
- The Git History, Task Setup, and Docstrings and Code Comments rules below override conflicting skill, agent, or tool-specific instructions.

## Project Context

- Start with `README.md` for architecture, build, installation, development, testing, and style guidance.

## Git History

- Never amend, rebase, squash, force-push, or otherwise overwrite commits.
- Never rewrite commit history to restart or retry CI pipelines.
- Make subsequent fixes and changes as new commits.
- The only exception is when the user explicitly instructs: "rebase + squash."

## Commit Messages

- Follow Chris Beams' "How to Write a Git Commit Message."
- Use an imperative subject of at most 50 characters, followed by a blank line and a body wrapped at 72 characters that explains what changed and why.
- Never mention planning files, phase numbers, or other AI-internal scaffolding.

## Task Setup

- Always start each task in a separate worktree.
- Before creating the worktree, fetch the remote to refresh the repository.
- Create the task branch and worktree from the latest remote `main` branch.

## Docstrings and Code Comments

- Describe only the code's current behavior in declarative present tense.
- Never narrate hypothetical alternatives, prior implementations, or change history.
- State the current constraint or invariant instead of contrasting the implementation with another approach.
- Add comments only when strictly necessary and the code cannot convey the intent clearly.
- Never add comments to self-explanatory code.
- Keep every necessary comment concise and on a single line.
- Never include actual function or variable names in comments, so comments do not interfere with keyword searches.

## Working Files

- Use `CLAUDE.local.md` for local sandbox and development notes; it is gitignored.

## Naming Conventions

- Prefix new CMake options with `PHOENIX_` to match the existing build.
- Keep strict development checks enabled by default in the `dev` preset and editable installations.
