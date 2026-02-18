# Phase 5: CI/CD - Context

**Gathered:** 2026-01-29
**Status:** Ready for planning

<domain>
## Phase Boundary

Automated testing and linting via GitHub Actions for the viscy-transforms monorepo package. Covers test workflow, lint workflow, and PR status checks. Documentation deployment is out of scope (deferred from Phase 4).

</domain>

<decisions>
## Implementation Decisions

### Workflow Triggers
- Run on push to main AND pull requests targeting main
- No path filtering for now (single package in monorepo)
- Cancel in-progress runs when new push to same branch
- Separate workflows: `test.yml` and `lint.yml`

### Test Matrix
- Python versions: 3.11, 3.12, 3.13 (full matrix)
- Operating systems: Ubuntu, macOS, Windows (full coverage)
- Cache uv dependencies using astral-sh/setup-uv caching
- Fail-fast enabled: cancel other jobs when one fails
- Include `alls-green` check job for required status checks:
  ```yaml
  check:
    name: All tests pass
    if: always()
    needs: [test]
    runs-on: ubuntu-latest
    steps:
      - uses: re-actors/alls-green@release/v1
        with:
          jobs: ${{ toJSON(needs) }}
  ```

### Lint Workflow
- Use `uvx prek` to run pre-commit hooks (matches local workflow)
- Single Python version: 3.13 (highest supported)
- Include `ruff format --check` for formatting enforcement
- Inline annotations enabled for lint errors on PR diffs

### PR Feedback
- All checks required to merge (test alls-green + lint)
- Coverage shown in CI logs via pytest flag (--cov with terminal output)
- No Codecov integration for now (future enhancement)

### Claude's Discretion
- Exact workflow file structure and job naming
- uv setup action version and configuration
- Timeout values for jobs

</decisions>

<specifics>
## Specific Ideas

- Use `re-actors/alls-green` action for the check job pattern
- Test command: `uv run pytest` (run all tests, no path filtering needed yet)
- Coverage via pytest-cov flag for terminal summary

</specifics>

<deferred>
## Deferred Ideas

- Codecov integration for coverage tracking over time
- Documentation deployment to GitHub Pages (Phase 4 deferred work)
- Path filtering when more packages are added to monorepo

</deferred>

---

*Phase: 05-ci-cd*
*Context gathered: 2026-01-29*
