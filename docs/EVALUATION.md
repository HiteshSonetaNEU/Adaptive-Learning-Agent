# Evaluation Guide

Run quick offline evaluation without the server:

```
python -m backend.scripts.evaluate --sessions 5 --steps 10 --out results
```

Outputs
- CSV summary in `results/` with per-session accuracy
- accuracy.png (if matplotlib available)

What to report
- Average accuracy and std across sessions/seeds
- Per-topic accuracy from balanced pools
- Optional: action distribution (count Q/PG actions)