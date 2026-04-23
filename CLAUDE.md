# VisCy — Claude Code Reference

## Project

VisCy is a **uv workspace monorepo** for virtual staining and computational microscopy. Sub-packages live under `packages/`.

## Repo Layout

```
pyproject.toml              # Root config (ruff, pytest, uv workspace)
packages/
  viscy-data/               # Data loading and Lightning DataModules
  viscy-models/             # Neural network architectures
  viscy-transforms/         # Image transforms
src/viscy/                  # Umbrella package (re-exports)
applications/               # Self-contained research applications
```

### Packages vs Applications

- **Shared code belongs in `packages/`**, not in applications.
- **Applications must not import from each other.** If two applications need the same logic, move it to an existing package or create a new one.
- Applications are consumers of packages — the dependency graph always flows `applications/ → packages/`, never sideways.

---

## Development

### Environment Setup

Use `uv` package manager. Run commands with `uv run <command>`. Edit `pyproject.toml` to modify dependencies and sync to update `uv.lock`.

```sh
uv venv -p 3.13
uv sync --all-packages --all-extras
```

If `uv` is not installed:
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On HPC, symlink the uv cache out of your home directory first:
```sh
mkdir -p /hpc/mydata/firstname.lastname/.cache/uv && ln -s /hpc/mydata/firstname.lastname/.cache/uv ~/.cache/uv
```

For full setup instructions (installing uv, creating a venv, syncing dependencies), see [CONTRIBUTING.md](./CONTRIBUTING.md).

### Common Commands

```sh
uvx ruff check packages/        # lint
uvx ruff check --fix packages/  # lint + auto-fix
uvx ruff format packages/       # format
uv run pytest                    # all tests
```

### Testing

```sh
uv run pytest                          # all tests
uv run pytest packages/viscy-data/     # single package (data)
uv run pytest packages/viscy-models/   # single package (models)
```

Prefer `{file}_test.py` in the same directory as `{file}.py`, unless there are import issues, in which case use `tests/`.

---

## Project Conventions

- Ruff config is centralized in the root `pyproject.toml` only. Sub-packages must NOT have their own `[tool.ruff.*]` sections. Ruff does not inherit config — any `[tool.ruff.*]` in a sub-package silently overrides the entire root config (including `lint.select`, `per-file-ignores`, etc.).
- Run `uvx prek run --files {files_you_edited}` (unless the change was simple) and fix typing and linting errors. Use `# type: ignore` as needed. The precommit will give you type errors which is useful — especially to know if you have incorrect code — but for many minor changes it's better to do this after testing. Use a subagent to apply complex fixes.

---

## Engineering Standards

### Git Workflow

- **NEVER** use `git commit --amend` or `git push --force` / `--force-with-lease` unless the user explicitly requests it. Always create NEW commits.
- ALWAYS use atomic commits: one logical change per commit. Never bundle unrelated changes.
- Never use `git add -A` or `git add .`. Always stage specific files by name.
- Always pull before pushing. If push is rejected, pull and retry — never force-push.

### Code Style

- Docstrings use **numpy style** (`convention = "numpy"`).
- Lint rules: `D, E, F, I, NPY, PD, W`.
- `D` rules are ignored in `**/tests/**` and notebooks.
- Format: double quotes, spaces, 120 char line length.
- Use a subagent to run tests and complex bash commands, especially those expected to return complex output.

#### Avoid Backwards Compatibility

In most cases it is incorrect to maintain backwards compatibility with a previous pipeline. This is a research codebase — changes are expected and encouraged. Keeping backwards compatibility risks MORE bugs, since someone can unknowingly run old code.

If you believe it is important to maintain backwards compatibility, explicitly ask the user if you should do so during the planning stage. If the user says no, then do not maintain backwards compatibility.

Delete and remove old code that is not used.

#### Use Context Managers for Resources

Always use context managers (`with` statements) when opening external resources like zarr stores, files, or database connections. Never assign them to a variable without a context manager — this leaks file handles and locks.

```python
# correct
with open_ome_zarr(path, mode="r") as plate:
    ...

# wrong — resource never closed
plate = open_ome_zarr(path, mode="r")
```

#### Prefer Raising Errors

Prefer raising errors instead of silently catching them. Errors are good and warn us of issues. For example, prefer `value = my_dictionary['key']` over `value = my_dictionary.get('key')` since the former will raise a `KeyError` to signal that the underlying data is not behaving as expected.

Only catch errors when there is a good reason to do so: for example, catching HTTP errors in order to retry a request.

If you find yourself writing an if statement, fallback, or except statement designed to avoid errors, ask yourself if it would be better to raise the error as a signal to the user.

#### Use Real Integration Tests

Tests should directly *import* the actual code we are trying to test. For example, if you are trying to test `my_function` on some sample data, your test should directly import `my_function` and run it on the sample data. Avoid testing "key behavior" or components in isolation when an integration test would catch more bugs.

Ask yourself if your test is actually covering the true function.

#### Imports

- Import at the top of the file. No inline imports without strong reason.
- Use absolute imports (`from packages.my_directory.my_file`) instead of relative.
- Do not modify `sys.path` for imports.

### Coding Philosophy

#### 1. Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

#### 2. Simplicity First

Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.
- Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

#### 3. Surgical Changes

Touch only what you must. Clean up only your own mess.

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: every changed line should trace directly to the user's request.

#### 4. Goal-Driven Execution

Define success criteria. Loop until verified.

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
1. [Step] → verify: [check]
2. [Step] → verify: [check]

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.
