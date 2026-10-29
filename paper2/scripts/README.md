# Online-arm runners (19 Oct 2026)

All need `.env` with `ANTHROPIC_API_KEY` in the project root and the Python 3.13 venv.

| script | what | cost (Haiku 5.5, thinking off) |
|---|---|---|
| `proxy_validation.sh` | answers all 1,986 LoCoMo questions on admit-all, structural gate (LOO weights, 50%) and random (50%); `PREFIX=online2` was the dated-facts pass | about $1 |
| `h1_mem0.sh` | Mem0 verbatim gate per conversation, then structural gate and random matched to its size, all answered; merges into `out/merged_h1_*` | about $1.5 |
| `proxy_analysis.py` | evidence-hit vs F1 tables; edit the three run names at the top for a different prefix | free |

Outputs of the 19 Oct runs: `out/online_*`, `out/online2_*`, `out/mem0_gate_conv-*`, `out/h1_*`, `out/merged_h1_*`;
analyses in `docs/proxy_pass1_analysis.txt` and `docs/proxy_pass2_analysis.txt`. The paper macros come from
`paper/make_numbers_online.py`.
