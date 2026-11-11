"""Sheaf-based contradiction detection for a small knowledge base.

Model
-----
World variables (the "global section" we want to recover):
  birth[e]      scalar, birth year of entity e
  city[e, t]    one-hot vector over CITIES, where e lived in year t

Every claim c is a local section: a stalk F(c) = R^{k_c}, a linear restriction
map A_c from the world space, and an observed value y_c. Stacking all claims
gives the coboundary-like map A: C^0 -> C^1 = (+)_c F(c).

  * y in im(A)            -> a global section exists (no contradiction)
  * y - proj_im(A)(y)     -> obstruction class; its norm measures inconsistency
  * ker(A)                -> gauge freedom (H^0 ambiguity), e.g. a group of people
                             only known relative to each other

Localization: robust fit  min_z  sum_c ||A_c z - y_c||_2  (group-L1, via IRLS);
claims with large residual are flagged as the likely-wrong ones.
"""

import itertools
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

CITIES = ["Seoul", "Busan", "Incheon", "Tokyo", "Osaka", "Paris", "Lyon"]
COUNTRY_OF = {"Seoul": "KR", "Busan": "KR", "Incheon": "KR",
              "Tokyo": "JP", "Osaka": "JP", "Paris": "FR", "Lyon": "FR"}
COUNTRIES = ["KR", "JP", "FR"]
YEARS = list(range(2010, 2020))
NC, NK, NT = len(CITIES), len(COUNTRIES), len(YEARS)

# city one-hot -> country one-hot (restriction map for country claims)
M_COUNTRY = np.zeros((NK, NC))
for i, c in enumerate(CITIES):
    M_COUNTRY[COUNTRIES.index(COUNTRY_OF[c]), i] = 1.0
# Same map rescaled to a partial isometry (M M^T = I), so that violating a country
# claim and violating the city claim it contradicts cost the same in the stalk metric.
COUNTRY_SIZE = M_COUNTRY.sum(1)
M_COUNTRY_ISO = M_COUNTRY / np.sqrt(COUNTRY_SIZE)[:, None]


# ---------------------------------------------------------------- data

def make_world(rng, n_anchor=14, n_relative=6):
    n = n_anchor + n_relative
    births = rng.integers(1950, 2001, size=n)
    residence = []
    for _ in range(n):
        cuts = sorted(rng.choice(range(1, NT), size=rng.integers(0, 3), replace=False))
        bounds = [0, *cuts, NT]
        cities = []
        for a, b in zip(bounds, bounds[1:]):
            cities += [rng.integers(NC)] * (b - a)
        residence.append(cities)
    return dict(n=n, n_anchor=n_anchor, births=births, residence=residence)


def make_claims(world, rng, corrupt_rate):
    n, na, births, res = world["n"], world["n_anchor"], world["births"], world["residence"]
    claims = []

    def offset():
        return int(rng.choice([-1, 1]) * rng.integers(3, 16))

    for e in range(n):
        if e < na:  # anchored entities: absolute birth info
            for _ in range(rng.integers(1, 3)):
                claims.append(dict(kind="birth", e=e, value=int(births[e])))
            for _ in range(rng.integers(1, 3)):
                t = int(rng.choice(YEARS))
                claims.append(dict(kind="age", e=e, t=t, value=t - int(births[e])))
        for _ in range(4):  # city intervals inside one true residence segment
            t0 = int(rng.integers(NT))
            t1 = t0
            while t1 + 1 < NT and res[e][t1 + 1] == res[e][t0] and rng.random() < 0.7:
                t1 += 1
            claims.append(dict(kind="city", e=e, t0=t0, t1=t1, value=int(res[e][t0])))
        for _ in range(3):
            ti = int(rng.integers(NT))
            claims.append(dict(kind="country", e=e, ti=ti,
                               value=COUNTRIES.index(COUNTRY_OF[CITIES[res[e][ti]]])))

    # relative claims: birth[a] - birth[b] = d. Relative-only entities form a
    # cycle-rich cluster with a single bridge to the anchored ones.
    rel = list(range(na, n))
    pairs = set()
    for a, b in itertools.combinations(rel, 2):
        if rng.random() < 0.6:
            pairs.add((a, b))
    pairs.add((int(rng.integers(na)), rel[0]))
    while len(pairs) < 30:
        a, b = sorted(rng.choice(n, 2, replace=False).tolist())
        pairs.add((a, b))
    for a, b in pairs:
        claims.append(dict(kind="reldiff", a=a, b=b, value=int(births[a] - births[b])))

    for c in claims:
        c["corrupt"] = False
        if rng.random() >= corrupt_rate:
            continue
        c["corrupt"] = True
        if c["kind"] in ("birth", "age", "reldiff"):
            c["value"] += offset()
        elif c["kind"] == "city":
            c["value"] = int(rng.choice([i for i in range(NC) if i != c["value"]]))
        else:
            c["value"] = int(rng.choice([i for i in range(NK) if i != c["value"]]))
    return claims


# ---------------------------------------------------------------- sheaf

def bidx(e):
    return e


def cidx(world, e, t):
    return world["n"] + (e * NT + t) * NC


def build_sheaf(world, claims, isometric=True, continuity=0.0):
    """Return per-claim (A_c, y_c) blocks with A_c acting on the world vector.
    continuity > 0 appends soft prior blocks continuity * (city[e,t] - city[e,t+1]) = 0
    after the claims ("people usually stay put"); they are never reported as flags."""
    dim = world["n"] + world["n"] * NT * NC
    blocks = []
    for c in claims:
        k = c["kind"]
        if k in ("birth", "age"):
            A = sp.csr_matrix(([1.0], ([0], [bidx(c["e"])])), shape=(1, dim))
            y = np.array([c["value"] if k == "birth" else c["t"] - c["value"]], float)
        elif k == "reldiff":
            A = sp.csr_matrix(([1.0, -1.0], ([0, 0], [bidx(c["a"]), bidx(c["b"])])), shape=(1, dim))
            y = np.array([c["value"]], float)
        elif k == "city":
            span = range(c["t0"], c["t1"] + 1)
            cols = [cidx(world, c["e"], t) + i for t in span for i in range(NC)]
            A = sp.csr_matrix((np.ones(len(cols)), (np.arange(len(cols)), cols)),
                              shape=(len(cols), dim))
            y = np.zeros(len(cols)); y[c["value"]::NC] = 1
        else:  # country: restriction city -> country
            o = cidx(world, c["e"], c["ti"])
            M = M_COUNTRY_ISO if isometric else M_COUNTRY
            A = sp.lil_matrix((NK, dim)); A[:, o:o + NC] = M; A = A.tocsr()
            y = np.zeros(NK); y[c["value"]] = M[c["value"]].max()
        blocks.append((A, y))
    for e in range(world["n"]) if continuity else ():
        for t in range(NT - 1):
            o0, o1 = cidx(world, e, t), cidx(world, e, t + 1)
            A = sp.csr_matrix((np.r_[np.full(NC, continuity), np.full(NC, -continuity)],
                               (np.r_[np.arange(NC), np.arange(NC)],
                                np.r_[np.arange(o0, o0 + NC), np.arange(o1, o1 + NC)])),
                              shape=(NC, dim))
            blocks.append((A, np.zeros(NC)))
    return blocks


def _stack(blocks):
    A = sp.vstack([b[0] for b in blocks]).tocsr()
    y = np.concatenate([b[1] for b in blocks])
    owner = np.repeat(np.arange(len(blocks)), [len(b[1]) for b in blocks])
    return A, y, owner


def _wls(A, y, w, ridge=1e-8):
    """Weighted least squares via sparse normal equations; the tiny ridge only
    picks a representative along ker(A) (unanchored / unconstrained variables)."""
    AtW = A.T.multiply(w).tocsr()
    return spsolve((AtW @ A + ridge * sp.eye(A.shape[1])).tocsc(), AtW @ y)


def obstruction_norm(blocks):
    """||y - proj_im(A) y||: size of the obstruction to a global section."""
    A, y, _ = _stack(blocks)
    return float(np.linalg.norm(A @ _wls(A, y, np.ones(len(y))) - y))


def robust_fit(blocks, iters=60, eps=1e-3, p=1.0, n_free=None):
    """IRLS for min_z sum_c ||A_c z - y_c||_2^p. Returns z and per-claim residual norms.
    p=1 is the convex group-L1 fit. p<1 runs the p=1 fit first and then continues with
    the nonconvex penalty from that starting point (closer to "count the violated
    claims"). Blocks from index n_free on (soft priors) always keep the p=1 penalty."""
    A, y, owner = _stack(blocks)
    n_free = len(blocks) if n_free is None else n_free
    w = np.ones(len(blocks))
    for q in ((1.0,) if p == 1.0 else (1.0, p)):
        for _ in range(iters):
            z = _wls(A, y, w[owner])
            r = np.sqrt(np.bincount(owner, (A @ z - y) ** 2, minlength=len(blocks)))
            w = 1.0 / np.maximum(r, eps)
            w[:n_free] **= 2.0 - q
    return z, r


def detect_sheaf(world, claims, thresh=0.3, isometric=True, continuity=0.0, p=1.0):
    """Flag claims with large robust residual. With isometric restriction maps an
    undecidable 1-vs-1 conflict splits its residual evenly (~0.5 each for one-hot
    stalks), so the threshold sits below that to flag both sides of a tie."""
    z, r = robust_fit(build_sheaf(world, claims, isometric, continuity), p=p, n_free=len(claims))
    return {i for i, ri in enumerate(r[:len(claims)]) if ri > thresh}, z


# ---------------------------------------------------------------- baselines

def _flag_minority(groups, flagged):
    """groups: key -> list of (claim_idx, value). Flag disagreeing minority (all on tie)."""
    for items in groups.values():
        vals = [v for _, v in items]
        if len(set(vals)) < 2:
            continue
        counts = {v: vals.count(v) for v in set(vals)}
        top = max(counts.values())
        modes = [v for v, k in counts.items() if k == top]
        for i, v in items:
            if len(modes) > 1 or v != modes[0]:
                flagged.add(i)


def detect_naive(world, claims):
    """Exact-key matching: compare only claims of the same kind about the same slot."""
    g, flagged = {}, set()
    for i, c in enumerate(claims):
        k = c["kind"]
        if k == "birth":
            g.setdefault(("birth", c["e"]), []).append((i, c["value"]))
        elif k == "age":
            g.setdefault(("age", c["e"], c["t"]), []).append((i, c["value"]))
        elif k == "reldiff":
            g.setdefault(("rel", c["a"], c["b"]), []).append((i, c["value"]))
        elif k == "city":
            for t in range(c["t0"], c["t1"] + 1):
                g.setdefault(("city", c["e"], t), []).append((i, c["value"]))
        else:
            g.setdefault(("country", c["e"], c["ti"]), []).append((i, c["value"]))
    _flag_minority(g, flagged)
    return flagged


def detect_rules(world, claims):
    """Hand-engineered rules: age->birth, city->country, reldiff checked against
    majority births when both endpoints are anchored. No cycle reasoning."""
    g, flagged = {}, set()
    for i, c in enumerate(claims):
        k = c["kind"]
        if k == "birth":
            g.setdefault(("birth", c["e"]), []).append((i, c["value"]))
        elif k == "age":
            g.setdefault(("birth", c["e"]), []).append((i, c["t"] - c["value"]))
        elif k == "city":
            for t in range(c["t0"], c["t1"] + 1):
                g.setdefault(("city", c["e"], t), []).append((i, c["value"]))
                g.setdefault(("country", c["e"], t), []).append(
                    (i, COUNTRIES.index(COUNTRY_OF[CITIES[c["value"]]])))
        elif k == "country":
            g.setdefault(("country", c["e"], c["ti"]), []).append((i, c["value"]))
    _flag_minority(g, flagged)

    est = {}
    for key, items in g.items():
        if key[0] != "birth":
            continue
        vals = [v for i, v in items if i not in flagged]
        if vals and len(set(vals)) == 1:
            est[key[1]] = vals[0]
    for i, c in enumerate(claims):
        if c["kind"] == "reldiff" and c["a"] in est and c["b"] in est:
            if est[c["a"]] - est[c["b"]] != c["value"]:
                flagged.add(i)
    return flagged


def _fundamental_cycles(n_nodes, edges):
    """edges: list of (u, v, d, claim_idx) meaning x_u - x_v = d.
    Returns [(discrepancy, [claim_idx, ...])] for each fundamental cycle of a BFS forest."""
    adj = [[] for _ in range(n_nodes)]
    for k, (u, v, d, _) in enumerate(edges):
        adj[u].append((v, k)); adj[v].append((u, k))
    pot, depth, parent = [None] * n_nodes, [0] * n_nodes, [None] * n_nodes
    tree = set()
    for s in range(n_nodes):
        if pot[s] is not None:
            continue
        pot[s], queue = 0, [s]
        while queue:
            u = queue.pop(0)
            for v, k in adj[u]:
                if pot[v] is None:
                    eu, ev, d, _ = edges[k]
                    pot[v] = pot[u] - d if eu == u else pot[u] + d
                    depth[v], parent[v] = depth[u] + 1, (u, k)
                    tree.add(k); queue.append(v)
    cycles = []
    for k, (u, v, d, idx) in enumerate(edges):
        if k in tree:
            continue
        path, a, b = [idx], u, v
        while a != b:
            if depth[a] < depth[b]:
                a, b = b, a
            a, kk = parent[a]
            path.append(edges[kk][3])
        cycles.append((pot[u] - pot[v] - d, path))
    return cycles


def detect_rules_cycles(world, claims):
    """`rules` for places, plus a greedy cycle-consistency check over all birth-type
    claims (birth/age as edges to a virtual root, reldiff as edges between people):
    repeatedly drop the claim lying on the most inconsistent fundamental cycles,
    as in loop-closure outlier rejection. Places are handled exactly like `rules`."""
    flagged = {i for i in detect_rules(world, claims)
               if claims[i]["kind"] in ("city", "country")}
    root = world["n"]
    edges = []
    for i, c in enumerate(claims):
        if c["kind"] == "birth":
            edges.append((c["e"], root, c["value"], i))
        elif c["kind"] == "age":
            edges.append((c["e"], root, c["t"] - c["value"], i))
        elif c["kind"] == "reldiff":
            edges.append((c["a"], c["b"], c["value"], i))
    return flagged | greedy_cycle_outliers(root + 1, edges)


def greedy_cycle_outliers(n_nodes, edges):
    """edges: [(u, v, d, idx)] meaning x_u - x_v = d. Greedy loop-closure-style
    outlier rejection; returns the set of idx judged wrong."""
    flagged, removed = set(), []
    while True:
        cycles = _fundamental_cycles(n_nodes, edges)
        bad, good = {}, {}
        for disc, path in cycles:
            tally = bad if disc != 0 else good
            for i in path:
                tally[i] = tally.get(i, 0) + 1
        if not bad:
            break
        worst = max(bad, key=lambda i: (bad[i], -good.get(i, 0)))
        removed.append(next(e for e in edges if e[3] == worst))
        edges = [e for e in edges if e[3] != worst]
    # cleanup: greedy removal over-deletes; re-admit claims consistent with the rest
    for e in removed:
        if all(d == 0 for d, _ in _fundamental_cycles(n_nodes, edges + [e])):
            edges.append(e)
        else:
            flagged.add(e[3])
    return flagged


# ---------------------------------------------------------------- eval

def prf(flagged, truth):
    tp = len(flagged & truth)
    p = tp / len(flagged) if flagged else 1.0
    r = tp / len(truth) if truth else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def run(seeds=range(20), rates=(0.05, 0.10, 0.20)):
    methods = {"naive": detect_naive, "rules": detect_rules, "rules+cyc": detect_rules_cycles,
               "raw@.5": lambda w, c: detect_sheaf(w, c, 0.5, isometric=False)[0],
               "iso@.3": lambda w, c: detect_sheaf(w, c, 0.3)[0],
               "iso+cont": lambda w, c: detect_sheaf(w, c, 0.3, continuity=0.7)[0]}
    kinds = ["birth", "age", "reldiff", "city", "country"]
    print(f"{'rate':>5} {'method':>9} | {'P':>5} {'R':>5} {'F1':>5} | recall by kind: "
          + " ".join(f"{k:>7}" for k in kinds))
    for rate in rates:
        agg = {m: [] for m in methods}
        byk = {m: {k: [0, 0] for k in kinds} for m in methods}
        for s in seeds:
            rng = np.random.default_rng(s)
            world = make_world(rng)
            claims = make_claims(world, rng, rate)
            truth = {i for i, c in enumerate(claims) if c["corrupt"]}
            for m, f in methods.items():
                fl = f(world, claims)
                agg[m].append(prf(fl, truth))
                for i in truth:
                    k = claims[i]["kind"]
                    byk[m][k][1] += 1
                    byk[m][k][0] += i in fl
        for m in methods:
            p, r, f1 = np.mean(agg[m], axis=0)
            rk = " ".join(f"{(a / b if b else float('nan')):7.2f}" for a, b in byk[m].values())
            print(f"{rate:5.2f} {m:>9} | {p:5.2f} {r:5.2f} {f1:5.2f} | {'':16}{rk}")
        print()


def run_obstruction(seed=0):
    """Obstruction norm should be 0 for clean data and grow with corruption."""
    print("corruptions  obstruction_norm")
    for rate in (0.0, 0.02, 0.05, 0.10, 0.20, 0.30):
        rng = np.random.default_rng(seed)
        world = make_world(rng)
        claims = make_claims(world, rng, rate)
        nc = sum(c["corrupt"] for c in claims)
        print(f"{nc:11d}  {obstruction_norm(build_sheaf(world, claims)):16.3f}")
    print()


def run_recovery(seeds=range(20), rate=0.10):
    """Accuracy of the recovered global section (anchored births, cities)."""
    ok_b = ok_c = tot_b = tot_c = 0
    for s in seeds:
        rng = np.random.default_rng(s)
        world = make_world(rng)
        claims = make_claims(world, rng, rate)
        _, z = detect_sheaf(world, claims)
        for e in range(world["n_anchor"]):
            tot_b += 1; ok_b += abs(z[bidx(e)] - world["births"][e]) < 0.5
        for e in range(world["n"]):
            for t in range(NT):
                v = z[cidx(world, e, t):cidx(world, e, t) + NC]
                if v.max() > 0.5:  # only slots the claims actually pin down
                    tot_c += 1; ok_c += v.argmax() == world["residence"][e][t]
    print(f"recovered section @ {rate:.0%} corruption: "
          f"birth acc {ok_b / tot_b:.3f} ({tot_b}), city acc {ok_c / tot_c:.3f} ({tot_c})\n")


if __name__ == "__main__":
    run_obstruction()
    run()
    run_recovery()
