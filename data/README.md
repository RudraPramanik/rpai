# Portfolio content (`backend/data`)

Phase 1 source of truth for later FastAPI (Phase 2) and RAG (Phase 4).

| Kind | Location | Use |
|---|---|---|
| **JSON facts** | `*.json` in this folder | UI + agent tools — **not** embedded for RAG |
| **Markdown knowledge** | `knowledge/**/*.md` | Semantic narratives for RAG |

## Canonical project inventory

Project slugs follow the **home** latest-projects list (`rp-frontend` `SectionMyLatestProject`) and the blueprint Phase-1 inventory:

`oms`, `laddu`, `admin-dashboard`, `bestenu`, `transform-portfolio-design-to-web-app-3`, `transform-portfolio-design-to-web-app-4`, `nike`, `resort`, `portfolio-web-design`

Do **not** treat `/project` page-only placeholders (e.g. `transform-portfolio-design-to-web-app-1`…`6`) as canonical knowledge projects. Phase 3 will reconcile that page to this inventory.

## Heading conventions (Markdown)

Baseline: `## Overview`, `## Problem`, `## Architecture`, `## Engineering Decisions`, `## Challenges`, `## Technologies`. Every file needs at least `## Overview`. Add AI-specific headings only when the project truly has that content.
