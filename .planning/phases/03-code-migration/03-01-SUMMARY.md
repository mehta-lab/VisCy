---
phase: 03-code-migration
plan: 01
subsystem: transforms
tags: [typing, torch, TypedDict, type-annotations]

# Dependency graph
requires:
  - phase: 02-package-structure
    provides: viscy-transforms package skeleton with __init__.py and py.typed
provides:
  - Type definitions for transform classes (Sample, ChannelMap, NormMeta)
  - Generic OneOrSeq type for channel flexibility
  - Normalization statistics types (LevelNormStats, ChannelNormStats)
  - HCS stack indexing (HCSStackIndex)
affects: [03-02-transforms-migration, viscy-data-future]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - TypedDict with total=False for optional fields
    - NamedTuple for immutable structured data
    - Type alias for complex nested types (NormMeta)
    - Generic TypeVar for single-or-sequence patterns

key-files:
  created:
    - packages/viscy-transforms/src/viscy_transforms/_typing.py
  modified: []

key-decisions:
  - "Extract only transform-relevant types from viscy.data.typing"
  - "Use NotRequired from typing_extensions for ChannelMap.target"
  - "Include explicit __all__ for controlled public API"

patterns-established:
  - "_typing.py: Internal module for package type definitions"
  - "TypedDict total=False: Allow optional fields in sample dictionaries"

# Metrics
duration: 4min
completed: 2026-01-28
---

# Phase 3 Plan 1: Type Definitions Summary

**Type definitions extracted from viscy.data.typing for standalone viscy-transforms package with Sample, ChannelMap, and normalization types**

## Performance

- **Duration:** 4 min
- **Started:** 2026-01-28T20:09:36Z
- **Completed:** 2026-01-28T20:13:36Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Created _typing.py with 7 essential type definitions
- All types pass type checking (ty check)
- Types match usage patterns in original _transforms.py

## Task Commits

Each task was committed atomically:

1. **Task 1: Fetch viscy.data.typing from GitHub and create _typing.py** - `bbb68f9` (feat)
2. **Task 2: Verify type compatibility with _transforms.py usage** - No commit (verification only)

## Files Created/Modified
- `packages/viscy-transforms/src/viscy_transforms/_typing.py` - Type definitions for transform classes (84 lines)

## Decisions Made
- Extracted only types used by transforms: Sample, ChannelMap, NormMeta, OneOrSeq, HCSStackIndex, LevelNormStats, ChannelNormStats
- Did NOT include dataset-specific types: DictTransform, SegmentationSample, TrackingIndex, TripletSample, AnnotationColumns, label constants
- Added docstrings to all types for clarity
- Included module-level docstring explaining extraction from viscy.data.typing

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Type foundation complete for _transforms.py migration
- Sample and ChannelMap types ready for import
- NormMeta type available for NormalizeSampled transform
- All types verified compatible with original usage patterns

---
*Phase: 03-code-migration*
*Completed: 2026-01-28*
