# Phase 3: Code Migration - Context

**Gathered:** 2026-01-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Migrate all 25 transform modules and their tests from the original VisCy monolith to the new `viscy-transforms` package structure. Clean imports working (`from viscy_transforms import X`), tests passing, old code removed.

</domain>

<decisions>
## Implementation Decisions

### Migration Strategy
- Big bang migration — move all 25 modules at once, no intermediate state
- Absolute imports between transform modules: `from viscy_transforms.normalize import X`
- Dependencies added via `uv add` — let uv resolve appropriate versions
- Treat as normal refactor — copy files as new, no special git history preservation

### Public API Design
- Flat imports from top-level: `from viscy_transforms import X` for all public transforms
- Explicit `__all__` at package level in `__init__.py`
- Explicit `__all__` at module level in each transform module
- Private utilities in `_utils.py` file (underscore prefix signals private)
- Include `py.typed` marker for type checking support
- No `__version__` attribute — use `importlib.metadata.version('viscy-transforms')` instead (decision from Phase 2)

### Test Organization
- Tests in `packages/viscy-transforms/tests/` — ship with the package
- Mirror source structure: `test_normalize.py` for `normalize.py`, etc.
- Package-level fixtures in `tests/conftest.py`
- Test data approach: follow existing VisCy test patterns

### Claude's Discretion
- Exact module migration order within big bang
- Dependency resolution details
- Test fixture implementation details based on existing VisCy patterns

</decisions>

<specifics>
## Specific Ideas

- Follow existing VisCy test data patterns (synthetic vs files) — analyze and match
- Integration test fixtures at workspace level are out of scope for this phase

</specifics>

<deferred>
## Deferred Ideas

- Workspace-level integration test fixtures — future phase when multiple packages exist

</deferred>

---

*Phase: 03-code-migration*
*Context gathered: 2026-01-28*
