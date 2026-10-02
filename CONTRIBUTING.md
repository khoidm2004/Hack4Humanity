# Contributing to Hack For Humanity

Thanks for your interest in contributing. This guide covers how to propose changes and how we format commit messages.

## Getting started

1. Fork the repository (if applicable) and clone it locally.
2. Create a branch from `main` for your change:
   ```bash
   git checkout -b feat/short-description
   ```
3. Make your changes, then commit using [Conventional Commits](#commit-messages).
4. Open a pull request against `main` and describe what you changed and why.

## Commit messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification. Every commit message should look like:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types

| Type       | When to use it                                      |
| ---------- | --------------------------------------------------- |
| `feat`     | A new feature                                       |
| `fix`      | A bug fix                                           |
| `docs`     | Documentation only                                  |
| `style`    | Formatting / whitespace (no code behavior change)   |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `perf`     | Performance improvement                             |
| `test`     | Adding or updating tests                            |
| `build`    | Build system or external dependencies               |
| `ci`       | CI configuration and scripts                        |
| `chore`    | Maintenance tasks that do not fit the above         |
| `revert`   | Reverts a previous commit                           |

### Scope (optional)

A scope narrows the change to an area of the codebase, for example:

```
feat(api): add volunteer registration endpoint
fix(ui): correct button alignment on mobile
docs(readme): clarify setup steps
```

### Description

- Use the imperative mood (“add”, not “added” or “adds”).
- Keep it short and focused on *what* changed.
- Do not capitalize the first letter after the colon.
- Do not end with a period.

### Breaking changes

Mark a breaking change in one of these ways:

1. Append `!` after the type/scope:
   ```
   feat(api)!: change response shape for /events
   ```
2. Or add a footer:
   ```
   BREAKING CHANGE: /events no longer returns nested venue objects
   ```

### Examples

```
feat: add event RSVP flow
fix(auth): handle expired session tokens
docs: update contributing guidelines
chore: add MIT license
refactor(db): simplify volunteer query helpers
```

## Pull requests

- Keep PRs focused on a single concern when possible.
- Reference related issues in the PR description.
- Ensure the commit history on the branch follows Conventional Commits.
- Be ready to update the PR based on review feedback.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
