# Domain Pitfalls: Python Monorepo Migration with uv Workspaces

**Domain:** Python monorepo migration (setuptools to hatchling, single package to uv workspace)
**Researched:** 2026-01-27
**Overall confidence:** MEDIUM-HIGH (verified against official docs and community issues)

---

## Critical Pitfalls

Mistakes that cause rewrites, major blockers, or architectural rework.

---

### Pitfall 1: Single requires-python Constraint Across Workspace

**What goes wrong:** uv workspaces enforce a single `requires-python` for the entire workspace, computed as the intersection of all members' values. If one package needs Python 3.11+ and another needs 3.12+, the workspace becomes 3.12+ only.

**Why it happens:** uv resolves dependencies for the entire workspace into a single lockfile. Different Python version constraints would make this impossible.

**Consequences:**
- Cannot test viscy-transforms on Python 3.11 if any future package requires 3.12
- Downstream users on 3.11 may be unable to install individual packages even if they'd work standalone

**Prevention:**
- Decide on the lowest Python version the workspace will support upfront (3.11 for VisCy)
- Document this constraint in workspace root pyproject.toml
- All packages must use `requires-python = ">=3.11"` (or the agreed floor)

**Detection:** `uv lock` will fail with Python constraint conflicts during resolution.

**Phase:** Address in Phase 1 (workspace scaffolding).

**Confidence:** HIGH (verified in [uv workspaces documentation](https://docs.astral.sh/uv/concepts/projects/workspaces/))

---

### Pitfall 2: Conflicting Dependencies Between Workspace Members

**What goes wrong:** All workspace members share a single lockfile. If viscy-transforms needs `numpy<2` and a future package needs `numpy>=2`, resolution fails.

**Why it happens:** uv workspaces assume all packages can coexist in one environment. This is by design for consistency but breaks when packages have incompatible dependency trees.

**Consequences:**
- Cannot lock the workspace
- Must either downgrade the newer package's requirements or remove the conflicting package from the workspace

**Prevention:**
- Survey dependency constraints before adding packages to workspace
- For PyTorch/NumPy heavy scientific packages, pin compatible version ranges early
- Use `[tool.uv.conflicts]` to declare mutually exclusive extras/groups if needed
- Consider path dependencies instead of workspace membership for packages with known conflicts

**Detection:** `uv lock` fails with dependency resolution errors.

**Phase:** Address during each package extraction (ongoing vigilance).

**Confidence:** HIGH (verified in [uv resolution docs](https://docs.astral.sh/uv/concepts/resolution/))

---

### Pitfall 3: uv-dynamic-versioning Does Not Work with uv Build Backend

**What goes wrong:** If you configure `build-backend = "uv"` instead of `build-backend = "hatchling.build"`, uv-dynamic-versioning silently fails or produces incorrect versions.

**Why it happens:** uv-dynamic-versioning is a hatchling plugin. It requires hatchling as the build backend to function.

**Consequences:**
- Packages built with version `0.0.0` or missing version
- CI releases fail
- PyPI uploads rejected or have wrong version

**Prevention:**
- Always use `build-backend = "hatchling.build"` when using uv-dynamic-versioning
- Never use `build-backend = "uv"` or `build-backend = "uv_build"` with this plugin
- Verify version in built wheel/sdist before publishing

**Detection:** Run `uv build` and check the generated filename for correct version.

**Phase:** Address in Phase 1 (build system setup).

**Confidence:** HIGH (verified in [uv-dynamic-versioning README](https://github.com/ninoseki/uv-dynamic-versioning))

---

### Pitfall 4: Import Leakage Between Workspace Members

**What goes wrong:** Python has no dependency isolation. viscy-transforms can accidentally import from viscy-data even if it doesn't declare that dependency, because all workspace members are installed in the same environment.

**Why it happens:** `uv sync` installs all workspace members as editable. They share a virtual environment. Python doesn't enforce import boundaries.

**Consequences:**
- Package works in monorepo but fails when installed standalone
- CI passes but users report `ModuleNotFoundError`
- Hidden coupling between packages

**Prevention:**
- Run `uv sync --package viscy-transforms` and test in isolation
- CI should test each package independently, not just the whole workspace
- Consider running `uv pip install viscy-transforms --no-deps` and verifying imports work with only declared dependencies

**Detection:** Install package in fresh venv outside workspace and run tests.

**Phase:** Address in Phase 1 (CI setup) and validate with each extraction.

**Confidence:** HIGH (explicitly documented in [uv workspaces docs](https://docs.astral.sh/uv/concepts/projects/workspaces/))

---

### Pitfall 5: Entry Points Lost During Migration

**What goes wrong:** Console script entry points (`viscy = "viscy.cli:main"`) stop working after migration from setuptools to hatchling.

**Why it happens:**
- Different configuration syntax between setuptools and hatchling
- Mixing `[project.scripts]` and `[options.entry_points]` in same file
- Forgetting to migrate entry points at all

**Consequences:**
- `viscy` CLI command not found after install
- Users cannot invoke tools from command line

**Prevention:**
- Explicitly audit all `[project.scripts]` and `[project.entry-points]` sections
- Test CLI commands work after migration: `uv run viscy --help`
- For VisCy: ensure `viscy = "viscy.cli:main"` is preserved in root package

**Detection:** After install, run the CLI command and verify it works.

**Phase:** Address when extracting any package with CLI entry points.

**Confidence:** MEDIUM-HIGH (multiple reports in [setuptools issues](https://github.com/pypa/setuptools/issues/4153))

---

## Moderate Pitfalls

Mistakes that cause delays, confusion, or technical debt (but are recoverable).

---

### Pitfall 6: Docker Build Inefficiency with Workspace Dependencies

**What goes wrong:** Docker builds copy entire workspace for every package, causing massive cache invalidation. Any change to any package rebuilds all Docker images.

**Why it happens:** uv requires workspace member files present to resolve dependencies. You can't just copy `uv.lock` and install third-party deps without the package sources.

**Prevention:**
- Use `uv sync --frozen --package <name>` which resolves from lockfile alone (partially works)
- Structure Dockerfiles to copy minimal files first, then add sources
- Consider `--no-editable` flag when available for self-contained builds

**Detection:** Docker build times remain high even for unrelated changes.

**Phase:** Address in Phase 4 (CI/CD optimization) or defer to later milestone.

**Confidence:** MEDIUM (discussed in [uv issue #6935](https://github.com/astral-sh/uv/issues/6935))

---

### Pitfall 7: IDE Workspace Recognition Failures

**What goes wrong:** VS Code/PyCharm don't understand uv workspace structure. Pylance shows import errors for valid workspace dependencies. Auto-complete fails.

**Why it happens:** IDEs look for standard Python project markers. uv workspaces use different conventions than Poetry/pip.

**Prevention:**
- Configure `.vscode/settings.json` with proper Python paths
- Use `uv sync` to ensure `.venv` exists with all packages installed
- May need to configure Pylance's `extraPaths` setting
- For PyCharm, mark `packages/*/src` as sources roots

**Detection:** IDE shows red squiggles on valid imports between workspace packages.

**Phase:** Address in Phase 1 (developer experience setup).

**Confidence:** MEDIUM (reported in [uv issue #10960](https://github.com/astral-sh/uv/issues/10960))

---

### Pitfall 8: Version File Not Updated in Editable Installs

**What goes wrong:** `viscy_transforms.__version__` returns stale value after git tag changes because editable installs don't rebuild version files.

**Why it happens:** uv-dynamic-versioning (and hatch-vcs) only update version files during build, not during editable development.

**Consequences:**
- Developers see wrong version locally
- CI may test with incorrect version metadata
- Version checks in code behave unexpectedly

**Prevention:**
- Don't rely on `__version__` in runtime code paths
- If needed, use `importlib.metadata.version("viscy-transforms")` instead
- Rebuild package before version-sensitive testing

**Detection:** `python -c "import viscy_transforms; print(viscy_transforms.__version__)"` shows old version.

**Phase:** Document in developer guide; not critical for Phase 1.

**Confidence:** MEDIUM (noted in [hatch-vcs documentation](https://pypi.org/project/hatch-vcs/))

---

### Pitfall 9: Namespace Package Migration Requires Simultaneous Update

**What goes wrong:** If VisCy kept `viscy.transforms` as a namespace package allowing both old and new code to coexist, all packages sharing that namespace must use identical `__init__.py` files.

**Why it happens:** Python's namespace package mechanism (pkgutil or pkg_resources style) requires consistency across all packages using the namespace.

**Consequences:**
- Import failures
- "Namespace package breaks module imports" errors
- Partial imports work, others fail randomly

**Prevention:**
- VisCy chose clean break approach (good decision)
- `from viscy_transforms import X` not `from viscy.transforms import X`
- No namespace packages, no coordination required

**Detection:** Would manifest as inconsistent import errors.

**Phase:** N/A for VisCy (already decided on clean break).

**Confidence:** HIGH (documented in [Python Packaging Guide](https://packaging.python.org/en/latest/guides/packaging-namespace-packages/))

---

### Pitfall 10: Optional Dependencies Conflicting Across Groups

**What goes wrong:** `viscy[dev]` includes `onnxruntime` but a future `viscy[gpu]` extra needs `onnxruntime-gpu`. These packages conflict and cannot be installed together.

**Why it happens:** PyPI packages can have mutually exclusive variants. uv resolves all extras together by default.

**Prevention:**
- Use `[tool.uv.conflicts]` to declare mutually exclusive extras:
  ```toml
  [tool.uv]
  conflicts = [
    [
      { extra = "cpu" },
      { extra = "gpu" },
    ]
  ]
  ```
- Document which extras can be combined
- Consider separate packages instead of conflicting extras

**Detection:** `uv lock` fails when conflicting extras exist without conflict declaration.

**Phase:** Address when defining extras for extracted packages.

**Confidence:** HIGH (documented in [uv dependencies docs](https://docs.astral.sh/uv/concepts/projects/dependencies/))

---

### Pitfall 11: CI Cache Explosion in Monorepos

**What goes wrong:** Every PR creates its own cache copy of `.venv` and `~/.cache/uv`. GitHub Actions storage fills up. Build times remain slow despite caching.

**Why it happens:** Default cache key includes PR-specific identifiers. No shared baseline cache.

**Prevention:**
- Use `uv cache prune --ci` before saving cache (removes pre-built wheels)
- Share cache across PRs using workflow-level cache key
- Consider weekly cache expiration
- Use `astral-sh/setup-uv` with smart cache management

**Detection:** GitHub Actions cache usage dashboard shows high storage; cache hit rate is low.

**Phase:** Address in Phase 4 (CI optimization).

**Confidence:** MEDIUM (discussed in [uv issue #2231](https://github.com/astral-sh/uv/issues/2231))

---

### Pitfall 12: GitHub Pages Jekyll Interference

**What goes wrong:** Sphinx documentation with `_static/` and `_templates/` directories doesn't deploy correctly to GitHub Pages.

**Why it happens:** Jekyll (GitHub Pages default) ignores directories starting with underscore.

**Prevention:**
- Add `.nojekyll` file to gh-pages branch root
- Configure GitHub Actions to create this file automatically
- Use `actions/deploy-pages` with proper configuration

**Detection:** CSS/JS missing on deployed docs site; 404 errors for `_static/` files.

**Phase:** Address in Phase 2 (documentation setup).

**Confidence:** HIGH (well-documented [Sphinx to GitHub Pages issue](https://lornajane.net/posts/2025/publish-to-github-pages-with-sphinx))

---

## Minor Pitfalls

Annoyances that are easily fixable but worth knowing about.

---

### Pitfall 13: Extra Name Normalization

**What goes wrong:** Extra named `foo_bar` must be installed as `pip install pkg[foo-bar]` (hyphen, not underscore).

**Why it happens:** PEP 503 normalizes package and extra names, converting underscores to hyphens.

**Prevention:**
- Use hyphens in extra names from the start
- Document the correct extra names in README

**Detection:** `pip install viscy[foo_bar]` fails with "extra not found".

**Phase:** Address during pyproject.toml authoring.

**Confidence:** HIGH (documented in [recursive optional dependencies article](https://hynek.me/articles/python-recursive-optional-dependencies/))

---

### Pitfall 14: src Layout Import Confusion

**What goes wrong:** Developer runs `python -c "import viscy_transforms"` from repo root and gets `ModuleNotFoundError`.

**Why it happens:** src layout requires package to be installed. Can't import directly from source tree.

**Prevention:**
- Always use `uv run python -c "..."` instead of bare `python`
- Document that `uv sync` is required before any testing
- This is intentional — prevents testing uninstalled code

**Detection:** Import errors when running Python directly instead of through uv.

**Phase:** Document in developer guide.

**Confidence:** HIGH (intentional design of src layout)

---

### Pitfall 15: Forgetting to Declare Workspace Dependencies

**What goes wrong:** Package A depends on Package B but forgets `{ workspace = true }` in sources. Resolution uses PyPI version instead of local version.

**Why it happens:** Easy to forget the explicit workspace source declaration.

**Prevention:**
- Template check: every inter-package dependency needs two parts:
  ```toml
  [project]
  dependencies = ["viscy-transforms>=0.1"]

  [tool.uv.sources]
  viscy-transforms = { workspace = true }
  ```
- CI should verify workspace packages resolve locally

**Detection:** `uv lock` downloads from PyPI instead of using local package.

**Phase:** Address with each package extraction.

**Confidence:** HIGH (documented in [uv workspaces docs](https://docs.astral.sh/uv/concepts/projects/workspaces/))

---

## Phase-Specific Warnings

| Phase | Topic | Likely Pitfall | Mitigation |
|-------|-------|----------------|------------|
| 1 | Workspace setup | requires-python intersection | Agree on floor (3.11) before starting |
| 1 | Build system | Wrong build backend | Use `hatchling.build`, never `uv` |
| 1 | Initial structure | Import leakage | Test packages in isolation from day 1 |
| 2 | Documentation | Jekyll blocking `_static` | Add `.nojekyll` to deploy workflow |
| 2 | API docs | Version not updating | Use importlib.metadata for runtime version |
| 3 | Package extraction | Entry points lost | Audit and test CLI commands |
| 3 | Dependencies | Conflicting numpy/torch versions | Pin compatible ranges early |
| 4 | CI/CD | Cache explosion | Use `uv cache prune --ci` |
| 4 | Docker | Full workspace copy | Use `--frozen --package` where possible |
| 5+ | Multiple extras | Conflicting extras | Declare conflicts in tool.uv |

---

## VisCy-Specific Risks

Based on the current pyproject.toml and PROJECT.md:

### Risk 1: PyTorch + NumPy Version Matrix

Current VisCy depends on `torch>=2.4.1` and `numpy` (unpinned). PyTorch has historically had tight NumPy version requirements. NumPy 2.0 migration is ongoing in the scientific Python ecosystem.

**Mitigation:**
- Pin NumPy range that's compatible with PyTorch 2.4+
- Test against NumPy 1.x and 2.x in CI
- Monitor [NumPy 2.0 ecosystem compatibility](https://github.com/numpy/numpy/issues/26191)

### Risk 2: MONAI + Kornia Dependency Conflicts

Both MONAI and Kornia have their own torch dependencies. Version mismatches can cause subtle runtime errors.

**Mitigation:**
- Run `uv tree` to inspect resolved versions
- Test with exact locked versions before release

### Risk 3: Clean Break Import Migration

Applications and examples will have broken imports (`from viscy.transforms import X`). This is intentional per PROJECT.md but creates temporary chaos.

**Mitigation:**
- Document the migration path clearly
- Consider a one-time deprecation notice release of old viscy that warns on import
- Update examples in a dedicated phase

---

## Sources

### Official Documentation (HIGH confidence)
- [uv Workspaces Documentation](https://docs.astral.sh/uv/concepts/projects/workspaces/)
- [uv Resolution Documentation](https://docs.astral.sh/uv/concepts/resolution/)
- [uv Dependencies Documentation](https://docs.astral.sh/uv/concepts/projects/dependencies/)
- [Python Packaging - Namespace Packages](https://packaging.python.org/en/latest/guides/packaging-namespace-packages/)
- [Sphinx Deploying Documentation](https://www.sphinx-doc.org/en/master/tutorial/deploying.html)

### GitHub Issues (MEDIUM-HIGH confidence)
- [uv #6935 - Workspaces and monorepo support](https://github.com/astral-sh/uv/issues/6935)
- [uv #10960 - Document best practices for monorepo](https://github.com/astral-sh/uv/issues/10960)
- [uv #6356 - Change-only testing in workspaces](https://github.com/astral-sh/uv/issues/6356)
- [setuptools #4153 - Entry points not installing](https://github.com/pypa/setuptools/issues/4153)

### Project-Specific (HIGH confidence)
- [uv-dynamic-versioning](https://github.com/ninoseki/uv-dynamic-versioning)
- [hatch-vcs](https://pypi.org/project/hatch-vcs/)
- [NumPy 2.0 Ecosystem Compatibility](https://github.com/numpy/numpy/issues/26191)

### Community Sources (MEDIUM confidence)
- [LlamaIndex Monorepo Overhaul](https://www.llamaindex.ai/blog/python-tooling-at-scale-llamaindex-s-monorepo-overhaul)
- [Attendi Python Monorepo Migration](https://attendi.nl/moving-all-our-python-code-to-a-monorepo-pytendi/)
- [Hynek - Recursive Optional Dependencies](https://hynek.me/articles/python-recursive-optional-dependencies/)
