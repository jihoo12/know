"""Experiment 5: sources.  Claims now come from sources of unequal quality.

  good    (5 sources)  3% random errors
  sloppy  (2 sources)  40% random errors
  biased  (1 source)   3% random errors, and every age it reports is +1
                       (a different convention, e.g. counting age: wrong as written,
                       but systematically so)

Two sheaf-native ways to use the source label:
  weights  the stalk of a claim from source s is scaled by w_s = logit(1 - eps_s),
           normalised so that eps = 5% gives 1 ("a lie from a reliable source costs
           more"). eps_s is learned by alternating fit <-> per-source flag rate.
  bias     one extra world variable b_s per source, used by its age claims
           (birth[e] - b_s = t - age), with a soft prior mu * b_s = 0. The convention
           becomes part of the global section: a learned transition map.
"""

import numpy as np
import scipy.sparse as sp

from sheafkb import (NT, NC, YEARS, make_world, make_claims, build_sheaf, robust_fit,
                     bidx, cidx, prf)

KINDS = ["good"] * 5 + ["sloppy"] * 2 + ["biased"]
ERR = {"good": 0.03, "sloppy": 0.40, "biased": 0.03}
NS = len(KINDS)
THRESH, CONT = 0.3, 0.7


def make_sourced_claims(world, rng):
    claims = make_claims(world, rng, 0.0)
    for c in claims:
        c["src"] = int(rng.integers(NS))
        kind = KINDS[c["src"]]
        c["biased"] = kind == "biased" and c["kind"] == "age"
        if c["biased"]:
            c["value"] += 1
        if rng.random() < ERR[kind]:
            c["corrupt"] = True
            if c["kind"] in ("birth", "age", "reldiff"):
                c["value"] += int(rng.choice([-1, 1]) * rng.integers(3, 16))
            elif c["kind"] == "city":
                c["value"] = int(rng.choice([i for i in range(NC) if i != c["value"]]))
            else:
                c["value"] = int(rng.choice([i for i in range(3) if i != c["value"]]))
    return claims


def weight_of(eps):
    eps = np.clip(eps, 0.01, 0.45)
    return np.log((1 - eps) / eps) / np.log(0.95 / 0.05)


def fit(world, claims, w_src=None, bias_mu=None):
    """Returns (flag set, z, per-source bias estimate or None)."""
    blocks = build_sheaf(world, claims, continuity=CONT)
    n = len(claims)
    w = np.ones(n) if w_src is None else w_src[[c["src"] for c in claims]]
    dim = blocks[0][0].shape[1]
    if bias_mu is not None:
        ext = []
        for i, (A, y) in enumerate(blocks):
            B = sp.lil_matrix((A.shape[0], NS))
            if i < n and claims[i]["kind"] == "age":
                B[0, claims[i]["src"]] = -1.0
            ext.append((sp.hstack([A, B]).tocsr(), y))
        for s in range(NS):
            ext.append((sp.csr_matrix(([bias_mu], ([0], [dim + s])), shape=(1, dim + NS)),
                        np.zeros(1)))
        blocks = ext
    blocks = [(A * w[i], y * w[i]) if i < n else (A, y) for i, (A, y) in enumerate(blocks)]
    z, r = robust_fit(blocks)
    flags = {i for i in range(n) if r[i] / w[i] > THRESH}
    return flags, z, (z[dim:dim + NS] if bias_mu is not None else None)


def fit_learned(world, claims, bias_mu=None, rounds=4):
    """Alternate: fit -> per-source flag rate (Laplace-smoothed) -> weights -> fit."""
    src = np.array([c["src"] for c in claims])
    w = np.ones(NS)
    for _ in range(rounds):
        flags, z, b = fit(world, claims, w, bias_mu)
        bad = np.bincount(src[sorted(flags)], minlength=NS) if flags else np.zeros(NS)
        eps = (bad + 1) / (np.bincount(src, minlength=NS) + 2)
        w = weight_of(eps)
    flags, z, b = fit(world, claims, w, bias_mu)
    return flags, z, b, eps


def score(world, claims, flags, z):
    rand = {i for i, c in enumerate(claims) if c["corrupt"]}
    bias = {i for i, c in enumerate(claims) if c["biased"] and not c["corrupt"]}
    p, r, f = prf(flags - bias, rand)          # flagged bias claims are not false alarms
    na = world["n_anchor"]
    birth = np.mean([abs(z[bidx(e)] - world["births"][e]) < 0.5 for e in range(na)])
    ok = tot = 0
    for e in range(world["n"]):
        for t in range(NT):
            v = z[cidx(world, e, t):cidx(world, e, t) + NC]
            if v.max() > 0.5:
                tot += 1; ok += v.argmax() == world["residence"][e][t]
    return [p, r, f, len(flags & bias) / max(len(bias), 1), birth, ok / tot]


def run(seeds=range(20)):
    true_w = weight_of(np.array([ERR[k] for k in KINDS]))
    methods = {
        "blind": lambda w, c: fit(w, c)[:2],
        "weights(learned)": lambda w, c: fit_learned(w, c)[:2],
        "weights(oracle)": lambda w, c: fit(w, c, true_w)[:2],
        "bias": lambda w, c: fit(w, c, bias_mu=0.3)[:2],
        "bias+weights(learned)": lambda w, c: fit_learned(w, c, bias_mu=0.3)[:2],
        "bias+weights(oracle)": lambda w, c: fit(w, c, true_w, bias_mu=0.3)[:2],
    }
    res = {m: [] for m in methods}
    eps_hat, b_hat, n_rand, n_bias, n_all = [], [], 0, 0, 0
    for s in seeds:
        rng = np.random.default_rng(s)
        world = make_world(rng)
        claims = make_sourced_claims(world, rng)
        n_all += len(claims); n_rand += sum(c["corrupt"] for c in claims)
        n_bias += sum(c["biased"] for c in claims)
        for m, f in methods.items():
            res[m].append(score(world, claims, *f(world, claims)))
        _, _, b, eps = fit_learned(world, claims, bias_mu=0.3)
        eps_hat.append(eps); b_hat.append(b)
    k = len(list(seeds))
    print(f"seeds {seeds[0]}-{seeds[-1]}: {n_all / k:.0f} claims/seed, "
          f"{n_rand / n_all:.1%} random errors, {n_bias / n_all:.1%} biased-convention claims")
    print(f"{'method':>22} |   P     R     F1  | bias-claims flagged | birth acc  city acc")
    for m, v in res.items():
        p, r, f, bf, ba, ca = np.mean(v, axis=0)
        print(f"{m:>22} | {p:.2f}  {r:.2f}  {f:.2f} | {bf:19.2f} | {ba:9.3f} {ca:9.3f}")
    eps_hat, b_hat = np.mean(eps_hat, 0), np.array(b_hat)
    print("learned per source (bias+weights):")
    print("  kind     " + " ".join(f"{x:>7}" for x in KINDS))
    print("  true eps " + " ".join(f"{ERR[x]:7.2f}" for x in KINDS))
    print("  eps_hat  " + " ".join(f"{x:7.2f}" for x in eps_hat))
    print("  b_hat    " + " ".join(f"{x:7.2f}" for x in b_hat.mean(0)))
    print(f"  biased source: b_hat within 0.1 of +1 in {np.mean(np.abs(b_hat[:, -1] - 1) < 0.1):.0%} "
          f"of seeds; any other source |b_hat| > 0.1 in "
          f"{np.mean(np.abs(b_hat[:, :-1]).max(1) > 0.1):.0%} of seeds\n")


if __name__ == "__main__":
    run(range(20))
    run(range(100, 120))
