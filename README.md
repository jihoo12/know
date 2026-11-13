# know

Finding the wrong claims in a small knowledge base by treating each claim as a local section of a sheaf.

A contradiction is then an *obstruction*: the claims can't be glued into one consistent world state. A robust (group-L1) fit puts the blame on the claims it has to violate most. Conversions (age to birth year), relative claims ("A is 3 years older than B"), time intervals, coarser facts (city to country), priors and source biases all become the same kind of object: a linear restriction map from the world state.

Write-up: **[Contradictions as Obstructions](https://jihoo12.github.io/papers/contradictions-as-obstructions/)**, an exploratory draft that hasn't been peer reviewed. The lab notebook behind it, with every experiment and finding, is [`NOTES.md`](NOTES.md) (in Korean).

## Findings in brief

- **Cycles come for free.** A contradiction that only appears around a cycle of relative claims is caught every time; hand-written rules miss most of them.
- **The stalk metric decides who gets blamed.** Restriction maps should make equally large lies cost the same. Otherwise the fit systematically blames the wrong side of a tie.
- **L1 recovery is a median vote on every cut.** The fit recovers the true errors exactly when, across every cut of the claim graph, the clean claims outvote the imbalance of the wrong ones. In practice the weakest point is the single least-supported fact.
- **L0 (consensus) is much stronger than L1**, and spanning-tree sampling guided by the L1 answer gets most of the way there in under 0.1 s per instance.
- **Source reliability can be learned without labels**, and a systematic bias (one source always reporting ages one year high) can be absorbed as a variable instead of being flagged as errors.

## Running

The only dependencies are NumPy and SciPy; `shell.nix` provides both. The seeds are fixed, so the output is deterministic.

| Command | What it runs | Time |
|---|---|---|
| `nix-shell --run 'python sheafkb.py'` | Experiments 1–3: obstruction norms, detection vs. baselines, recovered world state | ~20 s |
| `nix-shell --run 'python exp4_decoding.py'` | Experiment 4: cut-condition check, L1 / IRLS / exact L0 / greedy sweep, consensus | ~10 min (mostly the MILP) |
| `nix-shell --run 'python exp4_decoding.py consensus'` | Experiment 4, consensus tables only | ~1 min |
| `nix-shell --run 'python exp4_decoding.py quick'` | Experiment 4 on a few seeds, as a smoke test | ~2 min |
| `nix-shell --run 'python exp5_sources.py'` | Experiment 5: source reliability weights and bias variables | 1–2 min |

Without Nix, any Python 3 with `numpy` and `scipy` installed works the same: `python sheafkb.py`, and so on.

## Files

- `sheafkb.py`: the model (claims, restriction maps, obstruction norm, IRLS robust fit), the synthetic data, the baselines (`naive`, `rules`, `rules+cyc`) and experiments 1–3.
- `exp4_decoding.py`: the numeric-only decoding experiment (LP, MILP, greedy, RANSAC, L1-guided consensus, brute-force cut margins).
- `exp5_sources.py`: sources with learned reliability weights and per-source bias variables.
- `NOTES.md`: setup, results tables and findings for every experiment, including the ones that turned out wrong.

## Limitations

The data is synthetic and arrives already structured as claims; extracting claims from real text is not attempted. Categories are relaxed to real vectors, errors are assumed independent, and the strongest decoder (L1 + consensus) is so far implemented only for scalar claims. See the paper's limitations section and `NOTES.md` for more.

## License

Apache 2.0, see [`LICENSE`](LICENSE).
