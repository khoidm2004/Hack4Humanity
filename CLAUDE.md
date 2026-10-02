# Hack4Humanity

## Project
Hack For Humanity hackathon project. The repository currently contains only
scaffolding (`README.md`, `CONTRIBUTING.md`, `LICENSE`) — no application code,
tech stack, or dependency manifest exists yet.
**Fill in this section (what you're building, languages/frameworks/services)
once the project direction is set.**

## Commands
No test, lint, or build tooling exists yet — there is no `package.json`,
`pyproject.toml`, or `Makefile` in the repo. Agents must not guess or invent
commands. Once a toolchain is added, update this section with the exact
commands:
- Test: `<fill in>`
- Lint / type-check: `<fill in>`
- Build: `<fill in>`

## Conventions
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)
  (see `CONTRIBUTING.md`): `<type>[optional scope]: <description>`, imperative
  mood, no capital letter or trailing period after the colon.
- One branch per change, PRs opened against `main`.
- No code style, folder structure, or naming conventions have been
  established yet. **Add them here as they're decided** — don't let agents
  infer their own.
- Things Claude must never do: not yet specified. **Add project-specific
  "never touch" rules here as they come up.**

## Multi-agent workflow
- Write a task in `Artifacts/TASK.md` with a difficulty (`easy` | `medium` |
  `hard`) and run `/code-task`.
- Only the coordinator (the `/code-task` command, run in the main session —
  not a subagent) writes `Artifacts/state.json`.
- Only the tester — or the coordinator itself on `easy` tasks — appends to
  `Artifacts/dev-diary.md`.
- Agent definitions live in `.claude/agents/` (`planner`, `coder`, `tester`);
  the coordinator lives in `.claude/commands/code-task.md`.
- Routes: `easy` → planner → coder → done (no tester, no retries).
  `medium` / `hard` → planner → coder → tester → loop or finish.
- After 2 failed test runs in a row on `medium`/`hard`, the coordinator stops
  and asks you instead of retrying again.
- All agents and the coordinator treat this file as the source of truth for
  commands and conventions — they do not duplicate them elsewhere.
