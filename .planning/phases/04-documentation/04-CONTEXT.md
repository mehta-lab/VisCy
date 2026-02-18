# Phase 4: Documentation - Context

**Gathered:** 2026-01-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Zensical documentation deployed to GitHub Pages for the viscy workspace. This phase covers documentation tooling setup, API reference generation, and deployment — not tutorials, guides, or extensive prose content.

</domain>

<decisions>
## Implementation Decisions

### Documentation Structure
- Unified documentation site for entire workspace (not per-package sites)
- Landing page uses main repository README content
- Hybrid navigation: shared intro sections + package-specific API sections
- Getting Started section: leave empty for now (future phase)

### API Reference Depth
- Full reference level: parameters, return types, exceptions, cross-links
- Grouped by module (mirrors source structure: `_augment.py`, `_normalization.py`, etc.)
- Public API only — no internal/private APIs documented
- Type hints shown as inline signatures in function definitions

### Content Scope
- API documentation only for this phase
- Dual audience: researchers and ML engineers (accessible yet detailed)
- No tutorials or guides in this phase

### Claude's Discretion
- Zensical vs mkdocs-material choice (based on what works)
- Exact theme/styling choices
- Navigation depth and sidebar organization
- Build configuration details

</decisions>

<specifics>
## Specific Ideas

- Landing page should reflect the main repo README
- Package sections should be self-contained (viscy-transforms has its own API reference)
- Module-based grouping keeps mental model aligned with codebase

</specifics>

<deferred>
## Deferred Ideas

- **Reorganize batched_*.py into `batched/` submodule** — code structure change, add to backlog
- **Getting Started guide** — future phase after more packages exist
- **Tutorials and examples** — future phase
- **Contributor documentation** — future phase

</deferred>

---

*Phase: 04-documentation*
*Context gathered: 2026-01-28*
