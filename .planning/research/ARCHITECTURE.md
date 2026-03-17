# Architecture Patterns: uv Workspace Python Monorepo

**Domain:** Python monorepo for scientific imaging library
**Researched:** 2026-01-27
**Confidence:** HIGH (verified with official uv documentation)

## Recommended Architecture

```
viscy/                              # Repository root
├── pyproject.toml                  # Workspace root (virtual package)
├── uv.lock                         # Single lockfile for entire workspace
├── packages/                       # All extractable packages
│   ├── viscy-transforms/           # First extraction (this milestone)
│   │   ├── pyproject.toml          # Package config + dependencies
│   │   ├── src/
│   │   │   └── viscy_transforms/   # Import: from viscy_transforms import X
│   │   │       ├── __init__.py
│   │   │       └── *.py
│   │   └── tests/
│   │       └── test_*.py
│   ├── viscy-data/                 # Future package
│   ├── viscy-models/               # Future package
│   └── viscy-airtable/             # Future package
├── applications/                   # Publication code (kept, broken imports ok)
├── examples/                       # Usage examples (broken imports ok)
└── docs/                           # Zensical documentation
```

### Component Boundaries

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| **Workspace Root** | Defines workspace membership, shared tooling config | All packages (via `tool.uv.workspace`) |
| **viscy-transforms** | Image transformations (kornia, monai based) | Standalone, no workspace deps |
| **viscy-data** (future) | Data loading, HCS datasets | May depend on viscy-transforms |
| **viscy-models** (future) | Neural network architectures | May depend on viscy-transforms |
| **viscy-airtable** (future) | Airtable integration | May depend on viscy-data |
| **applications/** | Publication-specific pipelines | Not a package, imports from packages |
| **docs/** | Documentation site | References all packages |

### Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     WORKSPACE ROOT                               │
│  pyproject.toml: [tool.uv.workspace] members = ["packages/*"]   │
│  uv.lock: Single lockfile for ALL packages                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      packages/                                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐ │
│  │ viscy-transforms │  │   viscy-data     │  │ viscy-models  │ │
│  │   (standalone)   │  │ (depends on      │  │ (depends on   │ │
│  │                  │  │  transforms?)    │  │  transforms?) │ │
│  └──────────────────┘  └──────────────────┘  └───────────────┘ │
│           │                    │                    │           │
│           └────────────────────┼────────────────────┘           │
│                                ▼                                 │
│                    [tool.uv.sources]                            │
│           viscy-transforms = { workspace = true }               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              EXTERNAL CONSUMERS                                  │
│  pip install viscy-transforms                                   │
│  from viscy_transforms import RandGaussianSmoothd               │
└─────────────────────────────────────────────────────────────────┘
```

## Patterns to Follow

### Pattern 1: Virtual Workspace Root

**What:** The root `pyproject.toml` defines the workspace but is NOT itself a distributable package.

**When:** Monorepos where the root has no code to distribute, only workspace coordination.

**Why:** Prevents accidental attempts to install the root, clarifies that packages/ contains distributable code.

**Configuration:**
```toml
# Root pyproject.toml
[project]
name = "viscy-workspace"
version = "0.0.0"  # Not distributed
requires-python = ">=3.11"

[tool.uv]
# Makes this a virtual workspace root (not installable)
package = false

[tool.uv.workspace]
members = ["packages/*"]
```

### Pattern 2: Src Layout for Packages

**What:** Package source code lives in `packages/<name>/src/<import_name>/` not `packages/<name>/<import_name>/`.

**When:** Always for library packages.

**Why:** Prevents import confusion during development. Without src layout, `import viscy_transforms` might import local directory instead of installed package.

**Configuration:**
```toml
# packages/viscy-transforms/pyproject.toml
[build-system]
requires = ["hatchling", "uv-dynamic-versioning"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/viscy_transforms"]
```

### Pattern 3: Workspace Dependencies via Sources

**What:** Inter-package dependencies declared with `workspace = true`.

**When:** Package A depends on Package B, both in workspace.

**Why:** Ensures editable installs during development, proper resolution during publish.

**Configuration:**
```toml
# packages/viscy-data/pyproject.toml
[project]
dependencies = ["viscy-transforms"]

[tool.uv.sources]
viscy-transforms = { workspace = true }
```

### Pattern 4: Single Lockfile

**What:** One `uv.lock` at workspace root, none in packages.

**When:** Always.

**Why:** Ensures consistent dependency versions across all packages. `uv lock` operates on entire workspace.

**Commands:**
```bash
uv lock                           # Lock all packages
uv sync --package viscy-transforms # Sync specific package
uv run --package viscy-transforms pytest  # Run tests for package
```

### Pattern 5: Git-Based Versioning with uv-dynamic-versioning

**What:** Version derived from git tags, not hardcoded in pyproject.toml.

**When:** Libraries distributed to PyPI.

**Why:** Single source of truth for versions, automated release workflow.

**Configuration:**
```toml
[project]
name = "viscy-transforms"
dynamic = ["version"]

[build-system]
requires = ["hatchling", "uv-dynamic-versioning"]
build-backend = "hatchling.build"

[tool.hatch.version]
source = "uv-dynamic-versioning"

[tool.uv-dynamic-versioning]
vcs = "git"
style = "pep440"
# For monorepo: filter tags by package prefix
pattern = "^viscy-transforms-v(?P<version>.*)$"
```

### Pattern 6: PEP 735 Dependency Groups

**What:** Development dependencies in `[dependency-groups]` table, not optional-dependencies.

**When:** Test, dev, docs dependencies.

**Why:** PEP 735 standard, supported by uv, clear separation from runtime optional features.

**Configuration:**
```toml
[dependency-groups]
test = [
    "pytest>=8.0",
    "pytest-cov",
    "hypothesis",
]
dev = [
    { include-group = "test" },
    "pre-commit",
    "ruff",
]
```

### Pattern 7: Independent Package Testing

**What:** Each package has its own tests directory and can be tested in isolation.

**When:** Always.

**Why:** Validates package independence, faster CI, clearer ownership.

**Commands:**
```bash
# Test specific package
uv run --package viscy-transforms pytest packages/viscy-transforms/tests/

# Test all packages
uv run pytest
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: Flat Layout (No src/)

**What:** `packages/viscy-transforms/viscy_transforms/__init__.py`

**Why bad:** During development, Python may import local directory instead of installed package, hiding import errors that would appear for users.

**Instead:** Use src layout: `packages/viscy-transforms/src/viscy_transforms/__init__.py`

### Anti-Pattern 2: Per-Package Lockfiles

**What:** `packages/viscy-transforms/uv.lock`

**Why bad:** Breaks workspace benefits, dependency version conflicts between packages, CI complexity.

**Instead:** Single lockfile at workspace root.

### Anti-Pattern 3: Hardcoded Versions in pyproject.toml

**What:** `version = "0.1.0"` in pyproject.toml

**Why bad:** Manual version bumping, easy to forget, out of sync with git tags.

**Instead:** `dynamic = ["version"]` with uv-dynamic-versioning.

### Anti-Pattern 4: Root Package with Actual Code

**What:** Distributable code in workspace root alongside `[tool.uv.workspace]`.

**Why bad:** Confuses workspace coordination with package distribution, unclear responsibilities.

**Instead:** Virtual workspace root (`package = false`), all code in `packages/`.

### Anti-Pattern 5: Circular Dependencies Between Packages

**What:** viscy-transforms depends on viscy-data, viscy-data depends on viscy-transforms.

**Why bad:** Build order impossible, indicates poor separation of concerns.

**Instead:** Identify common code, extract to lower-level package, maintain DAG.

### Anti-Pattern 6: Mixing optional-dependencies and dependency-groups

**What:** Using optional-dependencies for dev/test dependencies.

**Why bad:** PEP 735 provides proper standard, optional-dependencies should be runtime features.

**Instead:** Use `[dependency-groups]` for dev/test, `[project.optional-dependencies]` for runtime features like `viscy-transforms[gpu]`.

## Build Order for Setup Tasks

Based on dependencies between setup tasks, recommended execution order:

```
Phase 1: Workspace Foundation (no dependencies)
├── Create root pyproject.toml with [tool.uv.workspace]
├── Configure ruff/pytest at workspace level
└── Create packages/ directory structure

Phase 2: First Package Scaffold (depends on Phase 1)
├── Create packages/viscy-transforms/pyproject.toml
├── Create src layout: packages/viscy-transforms/src/viscy_transforms/
└── Configure hatchling + uv-dynamic-versioning

Phase 3: Code Migration (depends on Phase 2)
├── Move viscy/transforms/*.py to packages/viscy-transforms/src/viscy_transforms/
├── Update internal imports (viscy.transforms → viscy_transforms)
└── Update __init__.py exports

Phase 4: Test Migration (depends on Phase 3)
├── Move tests/transforms/ to packages/viscy-transforms/tests/
├── Update test imports
└── Verify: uv run --package viscy-transforms pytest

Phase 5: Dependency Groups (depends on Phase 2)
├── Add [dependency-groups] to package pyproject.toml
├── Configure test/dev groups
└── Verify: uv sync --package viscy-transforms --group test

Phase 6: Dynamic Versioning (depends on Phase 2, can parallel Phase 3-5)
├── Configure uv-dynamic-versioning in pyproject.toml
├── Create git tag convention (viscy-transforms-v0.1.0)
└── Verify: uv build --package viscy-transforms

Phase 7: CI Updates (depends on Phases 4, 6)
├── Update GitHub Actions for monorepo testing
├── Configure package-specific test jobs
└── Add build/publish workflow
```

**Critical path:** Phase 1 → Phase 2 → Phase 3 → Phase 4 (testing validates migration)

**Parallelizable:** Phase 5 and Phase 6 can run alongside Phase 3-4 after Phase 2 completes.

## Scalability Considerations

| Concern | At 1 package | At 4 packages | At 10+ packages |
|---------|--------------|---------------|-----------------|
| Lock time | Fast (~2s) | Moderate (~10s) | Consider selective locking |
| Test time | Fast | Run per-package in CI | Parallel jobs essential |
| IDE support | Full | May need workspace config | Pylance workspace settings |
| Release | Single workflow | Per-package tags | Consider python-semantic-release |
| Dependency conflicts | Unlikely | Possible | Workspace-level pinning needed |

## Sources

**Official Documentation (HIGH confidence):**
- [uv Workspaces](https://docs.astral.sh/uv/concepts/projects/workspaces/) - Workspace configuration, members, sources
- [uv Project Init](https://docs.astral.sh/uv/concepts/projects/init/) - Project creation, --lib, --package flags
- [uv Project Config](https://docs.astral.sh/uv/concepts/projects/config/) - Configuration options
- [Hatch Build Configuration](https://hatch.pypa.io/latest/config/build/) - Hatchling src layout, packages

**PEP Standards (HIGH confidence):**
- [PEP 735 - Dependency Groups](https://peps.python.org/pep-0735/) - dependency-groups specification

**Tools (MEDIUM confidence - verified with repos):**
- [uv-dynamic-versioning](https://github.com/ninoseki/uv-dynamic-versioning) - Git-based versioning for hatchling
- [Python Developer Tooling Handbook](https://pydevtools.com/handbook/how-to/how-to-add-dynamic-versioning-to-uv-projects/) - Dynamic versioning guide

**Community Patterns (MEDIUM confidence):**
- [Python Workspaces (Monorepos)](https://tomasrepcik.dev/blog/2025/2025-10-26-python-workspaces/) - Real-world monorepo structure
- [Cracking the Python Monorepo](https://gafni.dev/blog/cracking-the-python-monorepo/) - Build patterns
- [uv Workspace Example Repo](https://github.com/mvoss02/uv_workspaces_example) - Reference implementation
