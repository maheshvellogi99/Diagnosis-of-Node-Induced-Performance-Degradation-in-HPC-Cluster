# Decision log

Copy of PLAN.md §16, kept current. PLAN.md is the source of truth.

| # | Date | Decision | Reason |
|---|---|---|---|
| D1 | 9 Oct 2026 | Layer 1 developed on Eagle; Layer 2 on PM100 | Eagle has no node list (F2); PM100 has `nodes` (F8) |
| D2 | 9 Oct 2026 | Per-node evidence is a **product** e-process, not an average | Average is a fixed-amount bet (F16) and cannot reach e-BH thresholds with ~1000 nodes |
| D3 | 9 Oct 2026 | Two conformal modes (E exact, A adaptive) | Exact sequential validity (F12) vs drift robustness |
| D4 | 9 Oct 2026 | Nagios used only for exploratory precursor analysis | Flagged nodes are offline (F9) |
| D5 | 9 Oct 2026 | Keep COMPLETED + TIMEOUT; drop other states | Follows F6 for cancelled/failed; others not runtime-meaningful |
| D6 | 9 Oct 2026 | Fixed Phase 1 feature set (no per-split search) | Cost; the paper's search needed 50,752 fits (F6) |
| D7 | 9 Oct 2026 | Mondrian group = user, fallback partition → global, n_min = 30 | User is the most predictive feature (F6); groups need enough history |
| D8 | 9 Oct 2026 | Loaders are lossless; filtering happens per phase | Every raw row is kept in `data/processed/` (null/0 allowed, negatives rejected); each phase applies and documents its own filters |
| D9 | 9 Oct 2026 | `state_raw` kept alongside `state` | Normalised `state` (COMPLETED/TIMEOUT/other) loses the original state; data cards and later filters need it |
