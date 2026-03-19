# Project Research Summary

**Project:** VisCy Modular Architecture (uv workspace monorepo)
**Domain:** Scientific Python package transformation (single package → workspace with independent subpackages)
**Researched:** 2026-01-27
**Confidence:** HIGH

## Executive Summary

This research evaluated transforming VisCy from a single setuptools-based package into a modern uv workspace monorepo with independently versioned subpackages. The recommended approach uses **hatchling + hatch-vcs + hatch-cada** for build and versioning, replacing the current setuptools + setuptools-scm configuration. The first extraction targets `viscy-transforms` (GPU augmentation transforms), establishing patterns for future extractions.

The monorepo approach enables users to install only what they need (`pip install viscy-transforms` instead of the entire VisCy stack), reduces dependency bloat, and allows independent release cycles. Critical to success: agreeing on a single Python version floor (3.11+) across all workspace members, preventing import leakage via isolated testing, and using src layout to avoid development-time import confusion. The architecture requires careful dependency management since all packages share a single lockfile.

Key risks center on the shared lockfile constraint (conflicting dependencies between packages break the workspace) and the clean break in import paths (`from viscy_transforms import X` not `from viscy.transforms import X`). Applications and examples will have temporarily broken imports until updated. Mitigation requires phase-gated extraction with validation at each step, comprehensive CI testing per package, and clear migration documentation for downstream users.

## Key Findings

### Recommended Stack

The stack modernizes VisCy's tooling while maintaining compatibility with the scientific Python ecosystem. Core shift: **hatchling replaces setuptools** because it's extensible via plugins and integrates cleanly with uv workspaces. Dynamic versioning uses **hatch-vcs (git tag based) + hatch-cada (workspace dependency rewriting)** instead of setuptools-scm. This combination enables independent package versioning with tags like `viscy-transforms@1.0.0` while correctly handling inter-package dependencies at build time.

**Core technologies:**
- **uv** (package manager, workspace orchestration) — 10-100x faster than pip, native workspace support, industry standard for 2025+
- **hatchling + hatch-vcs + hatch-cada** (build system + versioning) — Plugin-based extensibility; hatch-cada critical for workspace deps; hatch-vcs mature git versioning
- **Zensical** (documentation) — Official successor to Material for MkDocs (now in maintenance mode); 4-5x faster builds; Rust + Python
- **ruff** (linting + formatting) — Replaces black + isort + flake8; 200x faster; native Jupyter support; current project already uses it

**Note:** Cannot use uv's native build backend (`build-backend = "uv"`) because it doesn't support plugins yet. Hatchling required for hatch-vcs/hatch-cada functionality.

### Expected Features

**Must have (table stakes):**
- **Workspace member discovery** — `members = ["packages/*"]` glob pattern; users expect this to work
- **Shared lockfile** — Single `uv.lock` ensures reproducibility; standard uv workspace design
- **Editable inter-package dependencies** — `workspace = true` in sources; changes propagate immediately during development
- **Per-package testing** — `uv run --package viscy-transforms pytest` must work from any directory
- **Git-based versioning** — Dynamic version from VCS tags; avoids manual version bumps (scientific Python standard)
- **src layout** — `packages/*/src/*/` prevents import confusion; pytest/pip best practice
- **CI changed-package filtering** — Path-based filtering; don't test unchanged packages (monorepo efficiency requirement)

**Should have (competitive):**
- **Reusable CI workflows** — DRY GitHub Actions with parameters; reduces maintenance overhead
- **API documentation generation** — Auto-generated from docstrings (mkdocstrings/Zensical); keeps docs synchronized
- **Package-specific documentation** — Per-package docs with cross-linking; docs stay close to code
- **Parallel test execution** — pytest-xdist for faster runs; easy win for large test suites
- **Independent versioning** — Each package has own version via package-specific tags; good for loosely coupled packages

**Defer (v2+):**
- **Release automation** — python-semantic-release or manual workflow; manual releases acceptable initially
- **Coverage aggregation** — Combined coverage across packages; nice-to-have for later
- **Dependabot/Renovate** — Automated dependency updates; can add after initial stabilization
- **Dev containers** — Consistent environment via devcontainer.json; useful but not blocking

### Architecture Approach

The architecture uses a **virtual workspace root** (not a distributable package) that coordinates multiple independent packages under `packages/*`. Each package uses src layout (`packages/<name>/src/<import_name>/`) with its own pyproject.toml, preventing import confusion and enabling true independence. The single lockfile at workspace root ensures consistent dependency resolution, while workspace dependencies declared via `[tool.uv.sources]` with `workspace = true` enable editable development.

**Major components:**
1. **Workspace Root** — Defines membership via `[tool.uv.workspace]`, shared tooling config (ruff, mypy, pytest), not installable itself
2. **viscy-transforms (first extraction)** — Image transformations (kornia, monai based); standalone with no workspace dependencies
3. **viscy-data (future)** — Data loading, HCS datasets; may depend on viscy-transforms via workspace sources
4. **viscy-models (future)** — Neural network architectures; may depend on viscy-transforms
5. **applications/** — Publication code (not a package); broken imports acceptable during transition

**Critical path:** Workspace scaffolding → First package extraction → Code migration → Test migration. Phases 5+ (dependency groups) and 6 (dynamic versioning) can run in parallel after Phase 2 completes.

### Critical Pitfalls

1. **Single requires-python constraint** — uv enforces workspace-wide Python version intersection. If one package needs 3.12+, entire workspace becomes 3.12+. Users on 3.11 cannot install any package even if individually compatible. **Mitigation:** Agree on Python 3.11 floor upfront; document in workspace root; all packages must use `>=3.11`.

2. **Conflicting dependencies between members** — All packages share one lockfile. If viscy-transforms needs `numpy<2` and a future package needs `numpy>=2`, resolution fails and workspace cannot lock. **Mitigation:** Survey dependency constraints before adding packages; pin compatible ranges for PyTorch/NumPy early; consider path dependencies for genuinely incompatible packages.

3. **Import leakage between workspace members** — Python doesn't enforce dependency boundaries. viscy-transforms can accidentally import from viscy-data even without declaring it, because both are in the same environment. Works in monorepo, fails for users. **Mitigation:** Test each package in isolation (`uv sync --package <name>`); CI must test packages independently, not just whole workspace.

4. **uv-dynamic-versioning requires hatchling** — Using `build-backend = "uv"` breaks dynamic versioning; uv-dynamic-versioning is a hatchling plugin. **Mitigation:** Always use `build-backend = "hatchling.build"`; verify version in built wheel before publishing.

5. **Entry points lost during migration** — CLI commands (`viscy = "viscy.cli:main"`) stop working after setuptools → hatchling. Different config syntax; easy to forget. **Mitigation:** Audit all `[project.scripts]` sections; test CLI after migration: `uv run viscy --help`.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Workspace Foundation
**Rationale:** Establishes monorepo structure and prevents critical pitfalls (Python version floor, build system). Must come first because all packages depend on workspace configuration.

**Delivers:**
- Root pyproject.toml with `[tool.uv.workspace]` and `members = ["packages/*"]`
- Shared tooling config (ruff, mypy, pytest at workspace level)
- Python version floor decision (3.11+) documented
- Virtual workspace root (`package = false`)

**Addresses:**
- Workspace scaffolding (table stakes from FEATURES.md)
- Shared lockfile requirement
- Pre-commit/prek hooks for quality gates

**Avoids:**
- Pitfall #1 (Python version conflicts) by setting floor upfront
- Pitfall #4 (wrong build backend) by configuring hatchling immediately

### Phase 2: viscy-transforms Package Extraction
**Rationale:** First extraction establishes patterns for future packages. viscy-transforms chosen because it's standalone (no workspace dependencies), well-isolated, and delivers immediate user value.

**Delivers:**
- `packages/viscy-transforms/` with src layout
- Per-package pyproject.toml with hatchling + hatch-vcs + hatch-cada
- Git-based versioning configured (tag pattern: `viscy-transforms@X.Y.Z`)
- Clean import path: `from viscy_transforms import X`

**Uses:**
- hatchling build backend
- hatch-vcs for version from git tags
- hatch-cada for workspace dependency rewriting
- PEP 735 dependency groups for dev/test separation

**Implements:**
- src layout pattern from ARCHITECTURE.md
- Independent package testing workflow

**Avoids:**
- Pitfall #3 (import leakage) via isolated testing from day 1
- Pitfall #14 (src layout confusion) via clear documentation

### Phase 3: Code and Test Migration
**Rationale:** Moves actual code after scaffolding is validated. Separating this from Phase 2 allows validation of structure before content.

**Delivers:**
- Migrated code: `viscy/transforms/*.py` → `packages/viscy-transforms/src/viscy_transforms/`
- Updated imports within package
- Migrated tests: `tests/transforms/` → `packages/viscy-transforms/tests/`
- Updated test imports

**Addresses:**
- Test organization (table stakes)
- Clean import paths requirement

**Avoids:**
- Pitfall #5 (entry points lost) by auditing and testing CLI
- Pitfall #2 (dependency conflicts) via careful dependency specification

### Phase 4: CI/CD Updates
**Rationale:** CI must validate monorepo structure before merging. Comes after code migration so there's something to test.

**Delivers:**
- GitHub Actions workflows for monorepo testing
- Package-specific test jobs with path-based filtering
- Build verification (version correctness)
- Independent package testing validation

**Addresses:**
- CI changed-package filtering (table stakes)
- Path-based filtering requirement
- Build caching for efficiency

**Avoids:**
- Pitfall #11 (CI cache explosion) via `uv cache prune --ci`
- Pitfall #3 (import leakage) by testing packages in isolation

### Phase 5: Documentation Migration
**Rationale:** Documentation can be migrated after core functionality works. Zensical setup is independent of code migration.

**Delivers:**
- Zensical configuration replacing current docs
- Per-package documentation structure
- API documentation from docstrings (mkdocstrings)
- GitHub Pages deployment workflow

**Uses:**
- Zensical (successor to Material for MkDocs)
- mkdocstrings-python for API doc generation

**Addresses:**
- API documentation generation (differentiator)
- Package-specific documentation

**Avoids:**
- Pitfall #12 (Jekyll interference) via `.nojekyll` file in deploy workflow

### Phase 6: Validation and Documentation (Launch)
**Rationale:** Final validation before considering MVP complete. Documentation ensures future maintainers understand the patterns.

**Delivers:**
- Developer guide for monorepo workflow
- Migration guide for downstream users
- Example updates (fix broken imports in examples/)
- Version validation and test coverage verification

**Addresses:**
- Clean break import migration (VisCy-specific risk #3)
- Documentation of migration path

### Phase Ordering Rationale

- **Foundation first (Phase 1):** Workspace configuration is prerequisite for all packages; Python version floor prevents rework
- **Pattern establishment (Phase 2-3):** First extraction creates blueprint for future packages; validating structure before content prevents large-scale rework
- **Validation early (Phase 4):** CI must validate monorepo before considering it functional; testing in isolation catches import leakage
- **Documentation deferred (Phase 5):** Zensical setup independent of code migration; can proceed in parallel with Phase 4 if resources allow
- **Launch preparation (Phase 6):** User-facing docs and examples updated after core functionality proven

**Dependency ordering:**
- Phase 1 blocks all others (foundation)
- Phase 2 blocks Phase 3 (scaffolding before content)
- Phase 3 blocks Phase 4 (must have code to test)
- Phase 4 and Phase 5 can run in parallel after Phase 3
- Phase 6 depends on all previous phases

**Avoids pitfalls:**
- Phase-gated extraction prevents commitment to flawed structure
- Isolation testing at each phase catches import leakage early
- Build verification before merge prevents version issues in production

### Research Flags

**Phases needing deeper research during planning:**
- **Phase 2:** hatch-vcs tag pattern configuration (new as of v1.0.1, Jan 2026) — verify pattern syntax for monorepo
- **Phase 4:** GitHub Actions workspace testing patterns — sparse official guidance on monorepo path filtering
- **Phase 5:** Zensical migration from mkdocs.yml (Alpha software, v0.0.19) — may need fallback plan to mkdocs-material

**Phases with standard patterns (skip research-phase):**
- **Phase 1:** Workspace scaffolding — well-documented in uv official docs
- **Phase 3:** Code migration — standard Python refactoring patterns
- **Phase 6:** Documentation — standard technical writing

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Verified with official uv, hatchling, hatch-vcs docs; multiple successful deployments (pydantic-ai, MCP SDK) |
| Features | HIGH | Based on uv official workspace docs and PEP 735; table stakes well-established in community |
| Architecture | HIGH | Patterns verified in official uv documentation; src layout is pytest/pip best practice |
| Pitfalls | MEDIUM-HIGH | Critical pitfalls verified in official docs; moderate/minor based on GitHub issues and community reports |

**Overall confidence:** HIGH

Research is comprehensive with strong official documentation coverage. Lower confidence areas (Zensical, hatch-cada) have fallback options (mkdocs-material 9.7.0, uv-dynamic-versioning respectively) and don't block core functionality.

### Gaps to Address

**Gap: Zensical Alpha stability**
- **Impact:** Documentation generation may have bugs
- **Handling:** Keep mkdocs-material 9.7.0 as fallback; Zensical maintains compatibility with mkdocs.yml config
- **Validation:** Test Zensical during Phase 5 before committing; easy to roll back

**Gap: IDE workspace support**
- **Impact:** VS Code/PyCharm may not understand workspace structure; import errors shown for valid code
- **Handling:** Configure `.vscode/settings.json` with Python paths; use `uv sync` to populate `.venv`; document in developer guide
- **Validation:** Test with both VS Code and PyCharm during Phase 1

**Gap: PyTorch + NumPy version matrix**
- **Impact:** NumPy 2.0 migration ongoing; PyTorch has tight NumPy requirements; potential dependency conflicts
- **Handling:** Pin NumPy range compatible with PyTorch 2.4+; test against both NumPy 1.x and 2.x in CI matrix
- **Validation:** Run `uv tree` during Phase 2 to inspect resolved versions; monitor NumPy 2.0 ecosystem compatibility

**Gap: Docker build efficiency**
- **Impact:** Docker builds may copy entire workspace for every package; massive cache invalidation
- **Handling:** Defer to post-MVP; use `uv sync --frozen --package <name>` when available; structure Dockerfiles for minimal layer invalidation
- **Validation:** Measure Docker build times in CI during Phase 4; optimize if blocking

**Gap: Release automation**
- **Impact:** Manual release process initially; potential for version tag errors
- **Handling:** Document manual release workflow clearly; consider python-semantic-release post-MVP
- **Validation:** Test manual release workflow during Phase 6 with test PyPI

## Sources

### Primary (HIGH confidence)
- [uv Workspaces Documentation](https://docs.astral.sh/uv/concepts/projects/workspaces/) — Workspace configuration, member discovery, inter-package dependencies
- [uv Project Dependencies](https://docs.astral.sh/uv/concepts/projects/dependencies/) — Dependency groups (PEP 735), workspace sources
- [Hatchling Build Configuration](https://hatch.pypa.io/latest/config/build/) — Build backend, src layout, packages
- [PEP 735 - Dependency Groups](https://peps.python.org/pep-0735/) — dependency-groups specification
- [hatch-vcs PyPI](https://pypi.org/project/hatch-vcs/) — Git-based versioning for hatchling
- [hatch-cada GitHub](https://github.com/bilelomrani1/hatch-cada) — Workspace dependency rewriting at build time
- [Zensical Documentation](https://zensical.org/docs/get-started/) — MkDocs successor, setup and migration
- [pytest Good Integration Practices](https://docs.pytest.org/en/stable/explanation/goodpractices.html) — src layout, test organization

### Secondary (MEDIUM confidence)
- [Python Workspaces (Monorepos) - tomasrepcik.dev](https://tomasrepcik.dev/blog/2025/2025-10-26-python-workspaces/) — Real-world workspace structure patterns
- [uv Monorepo Best Practices Issue #10960](https://github.com/astral-sh/uv/issues/10960) — Community discussion on workspace patterns
- [LlamaIndex Monorepo Overhaul](https://www.llamaindex.ai/blog/python-tooling-at-scale-llamaindex-s-monorepo-overhaul) — Large-scale Python monorepo migration case study
- [Dynamic Versioning and Automated Releases](https://slhck.info/software/2025/10/01/dynamic-versioning-uv-projects.html) — uv-dynamic-versioning practical guide
- [Tweag Python Monorepo Guide](https://www.tweag.io/blog/2023-04-04-python-monorepo-1/) — Architectural patterns for Python monorepos
- [FOSDEM 2026 - Modern Python monorepo with uv](https://fosdem.org/2026/schedule/event/WE7NHM-modern-python-monorepo-apache-airflow/) — Apache Airflow's uv workspace migration

### Tertiary (LOW confidence, needs validation)
- [uv Issue #6935 - Workspaces and monorepo support](https://github.com/astral-sh/uv/issues/6935) — Docker build efficiency in workspaces
- [uv Issue #2231 - CI cache management](https://github.com/astral-sh/uv/issues/2231) — Cache pruning strategies
- [NumPy 2.0 Ecosystem Compatibility #26191](https://github.com/numpy/numpy/issues/26191) — NumPy version matrix tracking

---
*Research completed: 2026-01-27*
*Ready for roadmap: yes*
