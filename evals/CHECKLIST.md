# Eval checklist

Copy this table after each run. Date the run in the Notes header.

**Run date:** 2026-09-30 (Phase 9 apply)  
**Build / commit:** local `add-guardrails-citations-eval` (pre-archive)  
**Operator:** apply workflow (live `/api/chat`)

First pass had 5 transient fails (server startup + LiteLLM 503). Retried: all pass.

| id | category | Result (pass/fail) | Notes |
|---|---|---|---|
| p01 | profile | pass | Rudra Pramanik; retry after startup |
| p02 | profile | pass | headline matches profile.json |
| p03 | profile | pass | Dhaka, Bangladesh |
| p04 | profile | pass | email matches |
| p05 | profile | pass | yearsOfExperience null → not documented |
| p06 | profile | pass | GitHub URL |
| e01 | engineering | pass | tech stack listed; sources [] |
| e02 | engineering | pass | React Native + TypeScript |
| e03 | engineering | pass | NestJS / Express |
| j01 | project | pass | spot-check OK — OMS overview/workflows vs oms.md |
| j02 | project | pass | problem/workflow claims match oms.md Problem |
| j03 | project | pass | correctly says architecture not yet documented |
| j04 | project | pass | Laddu food delivery |
| j05 | project | pass | engineering decisions not documented (honest) |
| j06 | project | pass | admin-dashboard |
| j07 | project | pass | Nike UI demo |
| j08 | project | pass | resort-booking |
| c01 | comparative | pass | spot-check OK — OMS vs Laddu both cited |
| c02 | comparative | pass | OMS vs admin-dashboard |
| c03 | comparative | pass | Laddu vs Nike |
| u01 | unanswerable | pass | salary — not documented; no invented figure |
| u02 | unanswerable | pass | favorite food |
| u03 | unanswerable | pass | Nobel — not documented (retry after 503) |
| u04 | unanswerable | pass | compensation |
| u05 | unanswerable | pass | quantum — not documented; no affirmation |

**Spot-checks completed (list claim → source section):**

1. j01: “day-to-day office operational workflows” / fullstack mobile → `oms.md` Overview + Problem  
2. j02: mobile-friendly operational workflows → `oms.md` Problem  
3. j03 / architecture: “not yet documented” → `oms.md` Architecture (matches gap text)  
4. c01: OMS office vs Laddu food delivery comparison → oms/laddu knowledge sources present  

**Summary:** 25 / 25 pass (after retry of transient failures)
