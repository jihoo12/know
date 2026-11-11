"""Experiment 4: what limits the sheaf (L1) decoder?  L1 vs exact L0 vs greedy.

Numeric domain only: a connected d-regular graph on n people, every edge a claim
x_u - x_v = d_e, each corrupted independently with probability q.

  l1      min sum_e |x_u - x_v - d_e|          (the sheaf robust fit, solved exactly by LP)
  l0      min #violated claims                 (maximum consensus, exact MILP)
  greedy  loop-closure style cycle rejection   (sheafkb.greedy_cycle_outliers)

Cut criterion (LP duality + Hoffman's circulation theorem): the truth is an L1
minimiser iff for EVERY cut (S, S^c)
      | #cut errors pushing S up  -  #cut errors pushing S down |  <=  #clean cut edges,
i.e. L1 is a median vote across every cut. `cut_margin` checks this by brute force.
"""

import sys
import time
from math import comb
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

from sheafkb import greedy_cycle_outliers, robust_fit


def regular_graph(rng, n, d):
    """Connected simple d-regular graph: circulant start + random double edge swaps."""
    assert d % 2 == 0 and d < n
    while True:
        edges = {(i, (i + k) % n) if i < (i + k) % n else ((i + k) % n, i)
                 for i in range(n) for k in range(1, d // 2 + 1)}
        el = sorted(edges)
        for _ in range(20 * len(el)):
            i, j = rng.choice(len(el), 2, replace=False)
            (a, b), (c, e) = el[i], el[j]
            if rng.random() < 0.5:
                c, e = e, c
            n1, n2 = tuple(sorted((a, c))), tuple(sorted((b, e)))
            if a == c or b == e or n1 in edges or n2 in edges:
                continue
            edges -= {el[i], el[j]}; edges |= {n1, n2}
            el[i], el[j] = n1, n2
        seen, stack = {0}, [0]
        adj = {i: [] for i in range(n)}
        for u, v in el:
            adj[u].append(v); adj[v].append(u)
        while stack:
            for v in adj[stack.pop()]:
                if v not in seen:
                    seen.add(v); stack.append(v)
        if len(seen) == n:
            return np.array(sorted(el))


def instance(rng, n, d, q):
    E = regular_graph(rng, n, d)
    x = rng.integers(1950, 2001, size=n)
    err = np.where(rng.random(len(E)) < q,
                   rng.choice([-1, 1], len(E)) * rng.integers(3, 16, len(E)), 0)
    return E, x[E[:, 0]] - x[E[:, 1]] + err, err


def incidence(E, n):
    m = len(E)
    return sp.csr_matrix((np.r_[np.ones(m), -np.ones(m)],
                          (np.r_[np.arange(m), np.arange(m)], np.r_[E[:, 0], E[:, 1]])),
                         shape=(m, n))


def decode_l1(E, dv, n):
    m = len(E)
    B = incidence(E, n)
    I = sp.eye(m)
    res = linprog(np.r_[np.zeros(n), np.ones(m)],
                  A_ub=sp.vstack([sp.hstack([B, -I]), sp.hstack([-B, -I])]),
                  b_ub=np.r_[dv, -dv],
                  bounds=[(0, 0)] + [(None, None)] * (n - 1) + [(0, None)] * m, method="highs")
    return np.abs(B @ res.x[:n] - dv) > 0.5


def decode_irls(E, dv, n, p=1.0):
    """The sheaf code path (sheafkb.robust_fit). p=1 is L1; p<1 is the nonconvex variant."""
    B = incidence(E, n)
    _, r = robust_fit([(B[i], np.array([float(dv[i])])) for i in range(len(E))], p=p)
    return r > 0.5


def decode_l0(E, dv, n, time_limit=8.0):
    """Exact max-consensus by big-M MILP. Returns (flags, proven_optimal)."""
    m = len(E)
    B = incidence(E, n)
    R = 50.0                                   # births span 50 years, so |x_u - x_v| <= R
    M = sp.diags(R + np.abs(dv).astype(float))  # per-claim big-M, as tight as the range allows
    A = sp.vstack([sp.hstack([B, -M]), sp.hstack([-B, -M])])
    res = milp(np.r_[np.zeros(n), np.ones(m)],
               constraints=LinearConstraint(A, -np.inf, np.r_[dv, -dv]),
               integrality=np.r_[np.zeros(n), np.ones(m)],
               bounds=Bounds(np.r_[0, -R * np.ones(n - 1), np.zeros(m)],
                             np.r_[0, R * np.ones(n - 1), np.ones(m)]),
               options=dict(time_limit=time_limit))
    if res.x is None:
        return np.ones(m, bool), False
    return np.abs(B @ res.x[:n] - dv) > 0.5, res.status == 0


def decode_greedy(E, dv, n):
    fl = greedy_cycle_outliers(n, [(int(u), int(v), int(d), i)
                                   for i, ((u, v), d) in enumerate(zip(E, dv))])
    out = np.zeros(len(E), bool); out[list(fl)] = True
    return out


def decode_ransac(E, dv, n, rng, iters=300, guide=None):
    """Consensus by random spanning trees: a tree fixes every x_i, count the claims
    that disagree, keep the best. guide = per-claim penalty in [0, 1] added to the
    random tree-sampling key, e.g. 0.5 * (L1 flags): trees then prefer, but are not
    forced to use, the claims L1 believes. (Penalty 1 just reproduces the L1 answer.)"""
    m = len(E)
    best, best_bad = None, m + 1
    for _ in range(iters):
        key = rng.random(m) + (guide if guide is not None else 0)
        comp = list(range(n))
        adj = [[] for _ in range(n)]
        for k in np.argsort(key):
            u, v = int(E[k, 0]), int(E[k, 1])
            ru, rv = u, v
            while comp[ru] != ru: ru = comp[ru]
            while comp[rv] != rv: rv = comp[rv]
            if ru != rv:
                comp[ru] = rv
                adj[u].append((v, -dv[k])); adj[v].append((u, dv[k]))
        x = np.zeros(n); seen = [False] * n; seen[0] = True; stack = [0]
        while stack:
            u = stack.pop()
            for v, d in adj[u]:
                if not seen[v]:
                    seen[v] = True; x[v] = x[u] + d; stack.append(v)
        bad = x[E[:, 0]] - x[E[:, 1]] != dv
        if bad.sum() < best_bad:
            best, best_bad = bad, bad.sum()
    return best


_CUTS = {}


def cut_signs(E, n):
    """(2^(n-1)-1, m) int8 matrix: +1/-1 if the edge leaves/enters S, 0 if not cut."""
    if n not in _CUTS:
        ids = np.arange(1, 2 ** (n - 1))
        _CUTS[n] = ((ids[:, None] >> np.arange(n - 1)) & 1).astype(np.int8)  # node n-1 never in S
    S = np.c_[_CUTS[n], np.zeros(len(_CUTS[n]), np.int8)]
    return S[:, E[:, 0]] - S[:, E[:, 1]], S.sum(1)


def cut_margin(signs, err):
    """min over cuts of (#clean cut edges - |signed error imbalance|), and the argmin.
    > 0: truth is the unique L1 optimum;  < 0: L1 must fail;  = 0: tie."""
    sg = np.sign(err).astype(np.float32)
    s = signs.astype(np.float32)
    margin = np.abs(s) @ (sg == 0).astype(np.float32) - np.abs(s @ sg)
    k = int(margin.argmin())
    return margin[k], k


def vertex_cut_prediction(n, d, q):
    """P(every single-vertex cut passes the median test), vertices treated as independent."""
    ok = sum(comb(d, a) * comb(d - a, b) * (q / 2) ** (a + b) * (1 - q) ** (d - a - b)
             for a in range(d + 1) for b in range(d - a + 1) if abs(a - b) < d - a - b)
    return ok ** n


def f1(flag, truth):
    tp = (flag & truth).sum()
    if not flag.sum() or not truth.sum():
        return float(flag.sum() == truth.sum())
    p, r = tp / flag.sum(), tp / truth.sum()
    return 2 * p * r / (p + r) if tp else 0.0


def run_sweep(n=20, degrees=(4, 6, 10), rates=(0.05, 0.1, 0.15, 0.2, 0.3, 0.4), seeds=20):
    """exact = flagged set equals the true error set.
    l1 = LP, irls = the sheaf code path at p=1, p.5 = nonconvex IRLS (p=0.5),
    pred = vertex-cut prediction for l1, l0<k = L0 found a strictly smaller explanation
    than the truth (so the truth is not the maximum-consensus answer), l0opt = MILP proven
    optimal within the time limit."""
    print(f"n={n}, {seeds} seeds")
    print(f"{'deg':>3} {'q':>5} | exact-recovery rate {'':21} | mean F1 {'':24} | l0<k l0opt")
    names = ("l1", "irls", "p.5", "l0", "greedy")
    head = " ".join(f"{x:>6}" for x in names)
    print(f"{'':>9} | {head} {'pred':>5} | {head} |")
    for d in degrees:
        for q in rates:
            ex, fs = np.zeros(5), np.zeros(5)
            fewer = opt = 0
            for s in range(seeds):
                rng = np.random.default_rng(1000 * d + s)
                E, dv, err = instance(rng, n, d, q)
                truth = err != 0
                b, proven = decode_l0(E, dv, n)
                out = (decode_l1(E, dv, n), decode_irls(E, dv, n), decode_irls(E, dv, n, 0.5),
                       b, decode_greedy(E, dv, n))
                ex += [np.array_equal(x, truth) for x in out]
                fs += [f1(x, truth) for x in out]
                fewer += b.sum() < truth.sum(); opt += proven
            ex /= seeds; fs /= seeds
            print(f"{d:3d} {q:5.2f} | " + " ".join(f"{v:6.2f}" for v in ex)
                  + f" {vertex_cut_prediction(n, d, q):5.2f} | " + " ".join(f"{v:6.2f}" for v in fs)
                  + f" | {fewer / seeds:4.2f} {opt / seeds:5.2f}", flush=True)
        print()


def run_consensus(n=20, degrees=(4, 6, 10), rates=(0.05, 0.1, 0.15, 0.2, 0.3, 0.4), seeds=20,
                  seed_base=0, guide=0.5):
    """Cheap approximations of L0. seed_base=0 gives the same instances as run_sweep
    (guide strength 0.5 was picked there); any other value is a held-out set."""
    print(f"consensus by random spanning trees (300 trees), n={n}, seeds {seed_base}-"
          f"{seed_base + seeds - 1}, exact-recovery rate")
    print(f"{'deg':>3} {'q':>5} | {'l1':>6} {'ransac':>6} {'l1+ransac':>9} | P(clean tree)")
    for d in degrees:
        for q in rates:
            ex = np.zeros(3)
            for s in range(seed_base, seed_base + seeds):
                rng = np.random.default_rng(1000 * d + s)
                E, dv, err = instance(rng, n, d, q)
                truth = err != 0
                a = decode_l1(E, dv, n)
                r2 = np.random.default_rng(s)
                ex += [np.array_equal(x, truth) for x in
                       (a, decode_ransac(E, dv, n, r2), decode_ransac(E, dv, n, r2, guide=guide * a))]
            ex /= seeds
            print(f"{d:3d} {q:5.2f} | {ex[0]:6.2f} {ex[1]:6.2f} {ex[2]:9.2f} | {(1 - q) ** (n - 1):.3f}",
                  flush=True)
        print()


def run_cut_check(n=14, degrees=(4, 6), rates=(0.1, 0.2, 0.3, 0.4), seeds=100):
    """Verify the cut criterion against the LP, and see which cuts are the binding ones."""
    print(f"cut criterion vs LP, n={n} (all {2 ** (n - 1) - 1} cuts enumerated)")
    print(f"{'deg':>3} {'q':>5} | margin>0: n, l1 ok | margin<0: n, l1 ok | margin=0: n, l1 ok | "
          f"binding cut is 1 vertex (of margin<=0)")
    for d in degrees:
        for q in rates:
            cnt = {1: [0, 0], -1: [0, 0], 0: [0, 0]}
            single = 0
            for s in range(seeds):
                rng = np.random.default_rng(7000 * d + s)
                E, dv, err = instance(rng, n, d, q)
                signs, size = cut_signs(E, n)
                mg, k = cut_margin(signs, err)
                ok = np.array_equal(decode_l1(E, dv, n), err != 0)
                c = cnt[int(np.sign(mg))]
                c[0] += 1; c[1] += ok
                single += mg <= 0 and min(size[k], n - size[k]) == 1
            bad = cnt[-1][0] + cnt[0][0]
            print(f"{d:3d} {q:5.2f} | {cnt[1][0]:9d} {cnt[1][1]:6d}  | {cnt[-1][0]:9d} {cnt[-1][1]:6d}  | "
                  f"{cnt[0][0]:9d} {cnt[0][1]:6d}  | {single}/{bad}")
    print()


if __name__ == "__main__":
    t = time.time()
    if "consensus" in sys.argv:
        run_consensus(); run_consensus(seed_base=500, seeds=50)
    elif "quick" in sys.argv:
        run_cut_check(seeds=10); run_sweep(seeds=3)
    else:
        run_cut_check(); run_sweep(); run_consensus(); run_consensus(seed_base=500, seeds=50)
    print(f"({time.time() - t:.0f}s)")
