"""Phase C: certificates, covariance bounds and radii on the real frozen exports (brief §9 Phase C, with the
settings of the Phase A review).

Per dataset/snapshot (certify.model.from_export, decisions D2(a)-(c)):
  oracle      xbar and e = H^-1 g (g = H x - b) by the landmark-eliminated global solve with extended-
              precision refinement; Sigma_ii by the dense inverse of the reduced (pose/marker) system and the
              landmark formula (double precision; any flagged covariance violation is re-verified per block by
              an extended-precision refined solve with the unit columns of the block).
  C1, C2      certify.certs (C2 not on KITTI full, per the settings).
  C3          certify.certs.patch_pass at the requested K; on KITTI full K = 3 is used on 200 uniformly
              sampled poses with K = 2 patches everywhere else (Theorem 3.5 allows any patch per node).
  C4 + hubs   certify.hubs: ChArUco S = all markers jointly with the whole graph as hub window (Shat_loc =
              Shat, theta_S exact); kitti06-prefix w = 10 with hubs one at a time (certify/hubs.py docstring).
  bounds      trivial / patch / Krylov (l = 3, 5) lower bounds, cut and hub-aware upper bounds, radii
              (delta = 0.05) and the stopping flag E <= eta S_lo (eta = 0.1) with E the smallest valid
              certificate in the row.
Outputs go to results/phaseC/<dataset>/.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

from certify import certs, covbounds, hubs, radii
from certify import chain as ch
from certify import model as mdl
from certify.linalg import GlobalMarginals, SPDSystem

REPO = Path(__file__).resolve().parents[1]
E = REPO / "exports"
OUT = REPO / "results" / "phaseC"
ETA, DELTA = radii.ETA, radii.DELTA
TOL = 1e-9

CHARUCO = dict(C2=True, C3=(1, 2, 3), C4_w=(1, 3), C4_K=(1, 2, 3), hubs="joint_markers", krylov="all", p_L=None)
PREFIX = dict(C2=True, C3=(1, 2, 3), C4_w=(10,), C4_K=(1, 2, 3), hubs="one_at_a_time", krylov="poses+sample", p_L=1e-6)
FULL = dict(C2=False, C3=(1, 2), C3_K3_sample=200, C4_w=(), C4_K=(), hubs=None, krylov="poses+sample", p_L=1e-6)
DATASETS = {
    "charuco_islam_025": (E / "charuco_islam/frac025", CHARUCO),
    "charuco_islam_050": (E / "charuco_islam/frac050", CHARUCO),
    "charuco_islam_100": (E / "charuco_islam/frac100", CHARUCO),
    "charuco_ba_init": (E / "charuco_ba/init", CHARUCO),
    "charuco_ba_end": (E / "charuco_ba/end", CHARUCO),
    "kitti06_prefix_conv": (E / "kitti06_prefix161/conv", PREFIX),
    "kitti06_prefix_init": (E / "kitti06_prefix161/init", PREFIX),
    "kitti06_full_conv": (E / "kitti06_full/conv_noanchor", FULL),
}
A5 = (1, 2, 5, 10, 20, 50, 100)
CURVE_EXTRA = (3, 7, 15, 30, 70, 150, 200, 300, 500, 700, 1000, 1500, 2000, 2500, 3000)
KRYLOV_LANDMARK_SAMPLE = 2000


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ------------------------------------------------------------------------------------------ helpers
def eval_iterations(iters_avail, t_floor):
    want = set(A5) | set(CURVE_EXTRA)
    if t_floor is not None and np.isfinite(t_floor):
        want.add(int(t_floor))
    want.add(int(iters_avail[-1]))
    return [int(t) for t in sorted(want) if t in set(iters_avail.tolist())]


def load_states(model, info, iters):
    ex = info["export"]
    kept = info["kept_export_blocks"]
    Z = ex.states()
    avail = np.load(ex.path / "states_iters.npy")
    row = {int(t): k for k, t in enumerate(avail)}
    cols = np.concatenate([np.arange(ex.offsets[b], ex.offsets[b] + ex.dims[b]) for b in kept])
    return np.stack([np.asarray(Z[row[t]])[cols] for t in iters], axis=1)


def verify_cov_block(model, sysm, i):
    """Sigma_ii by an extended-precision refined solve with the unit columns of block i."""
    d = int(model.dims[i])
    Eb = np.zeros((model.N, d))
    s = model.block_slice(i)
    Eb[s] = np.eye(d)
    X = sysm.solve(Eb, refine=3, extended=True)
    S = X[s]
    return 0.5 * (S + S.T)


def mu_pair(C, X):
    if X is None:
        return np.nan, np.nan
    w = np.linalg.eigvalsh(0.5 * (C.T @ X @ C + (C.T @ X @ C).T))
    return float(w[0]), float(w[-1])


# ------------------------------------------------------------------------------------------ main
def run_dataset(name, workers=16, only=None):
    path, cfg = DATASETS[name]
    out_dir = OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    timing = {}
    t_all = time.perf_counter()
    model, info = mdl.from_export(path, p_L=cfg["p_L"])
    n = model.n_blocks
    log("%s: %d blocks, %d dof" % (name, n, model.N))
    d2 = json.loads((OUT / "d2" / ("%s.json" % name)).read_text())
    mrec = [m for m in d2["models"] if m["p_L"] == cfg["p_L"]][0]
    pose_floor = float(mrec["pose_floor"]["floor"])
    t_floor = mrec["pose_floor"]["t_floor"]
    iters_avail = np.load(Path(path) / "states_iters.npy")
    iters = eval_iterations(iters_avail, t_floor)
    log("iterations: %s" % iters)

    # ---------------------------------------------------------------- oracle
    t0 = time.perf_counter()
    sysm = SPDSystem(model.H, model.dims, model.elim, refine=3, K_lo=model.H_lo)
    assert sysm.ok
    xbar = sysm.solve(model.b)
    X = load_states(model, info, iters)
    G = model.residual(X)
    Etrue = sysm.solve(G)
    met = certs.Metric(model)
    err = met.norms(Etrue)                       # n x t
    err_xdiff = met.norms(X - xbar[:, None])     # cross-check
    gm = GlobalMarginals(sysm)
    Sig = [gm.block(int(model.offsets[i]), int(model.dims[i])) for i in range(n)]
    timing["oracle_s"] = time.perf_counter() - t0
    log("oracle done (%.1fs); max |err(H^-1 g) - err(x - xbar)| / err = %.2e" % (
        timing["oracle_s"], np.nanmax(np.abs(err - err_xdiff) / np.maximum(err, 1e-300))))
    lam = np.array([mu_pair(met.C[i], Sig[i]) for i in range(n)])  # oracle lambda_min, lambda_max

    res = dict(name=name, n_blocks=n, n_dof=model.N, iters=iters, pose_floor=pose_floor, t_floor=t_floor,
               counts={k: info.get(k) for k in ("single_edge_eliminated", "single_edge_eliminated_by_kind",
                                                "zero_information_excluded", "landmark_priors")})
    cols = {}  # per-row arrays, keyed; rows = (block, iteration) for iteration-dependent quantities

    # ---------------------------------------------------------------- C1
    t0 = time.perf_counter()
    r1 = certs.c1(model, met, G)
    timing["C1_s"] = time.perf_counter() - t0
    res["rho_M"], res["rho_M_upper"] = r1["rho"], r1["rho_upper"]
    res["C1_iterations"] = r1["iterations"]
    cols["E_C1"] = r1["E"] if r1["E"] is not None else np.full_like(err, np.nan)
    log("C1: rho(M) = %.4f (upper %.4f), %.1fs" % (r1["rho"], r1["rho_upper"], timing["C1_s"]))
    onestep = dict(C1=(r1["M"], r1["nu"]))
    del r1

    # ---------------------------------------------------------------- C2
    if cfg["C2"]:
        t0 = time.perf_counter()
        r2 = certs.c2(model, met, G)
        timing["C2_s"] = time.perf_counter() - t0
        res["rho_Mhat"], res["rho_Mhat_upper"] = r2["rho"], r2["rho_upper"]
        cols["E_C2"] = r2["E"] if r2["E"] is not None else np.full_like(err, np.nan)
        log("C2: rho(Mhat) = %.4f (upper %.4f), %.1fs" % (r2["rho"], r2["rho_upper"], timing["C2_s"]))
        onestep["C2"] = (r2["M"], r2["nu"])
        del r2
    else:
        cols["E_C2"] = np.full_like(err, np.nan)

    # ---------------------------------------------------------------- C3 + patch/cut bounds
    c3 = {}
    keep_P2 = None
    for K in cfg["C3"]:
        t0 = time.perf_counter()
        P = certs.patch_pass(model, met, K, G, workers=workers, chunk=32)
        if K == 2 and cfg.get("C3_K3_sample"):
            keep_P2 = P
        timing["C3_K%d_patches_s" % K] = time.perf_counter() - t0
        r3 = certs.c3_from_patches(model, P)
        c3[K] = dict(E=r3["E"], s=r3["s"], A=r3["A"], rho=r3["rho"], rho_upper=r3["rho_upper"], iterations=r3["iterations"],
                     t_global=r3["t_global"], row_sum_max=r3["row_sum_max"],
                     inv_ii=[p["inv_ii"] for p in P], cut_ii=[p["cut_ii"] for p in P],
                     certifiable=np.array([bool(p["certifiable"]) for p in P]),
                     sizes=np.array([[p["n_patch"], p["n_patch_dofs"], p["n_boundary"], p["n_exterior"]] for p in P]),
                     t_block=np.array([[p["t_local"], p["t_coeff"], p["t_cut"]] for p in P]))
        log("C3 K=%d: rho(A) = %.4f (upper %.4f), patches %.1fs, global %.2fs" % (K, r3["rho"], r3["rho_upper"],
                                                                                  timing["C3_K%d_patches_s" % K], r3["t_global"]))
        c3[K]["patches_raw"] = P if K == 2 and cfg.get("C3_K3_sample") else None
        del r3
    if cfg.get("C3_K3_sample"):
        rng = np.random.default_rng(0)
        poses = np.flatnonzero(model.kinds == "pose")
        samp = np.sort(rng.choice(poses, size=min(cfg["C3_K3_sample"], len(poses)), replace=False))
        t0 = time.perf_counter()
        P3 = certs.patch_pass(model, met, 3, G, blocks=samp, workers=workers, chunk=4)
        timing["C3_K3_sample_patches_s"] = time.perf_counter() - t0
        # mixed patches: K = 3 for the sample, K = 2 elsewhere (Theorem 3.5 allows any patch per node)
        in_s = np.zeros(n, bool)
        in_s[samp] = True
        allP = [p for p in keep_P2 if not in_s[p["i"]]] + P3
        r3m = certs.c3_from_patches(model, allP)
        c3["3mix"] = dict(E=r3m["E"], s=r3m["s"], A=r3m["A"], rho=r3m["rho"], rho_upper=r3m["rho_upper"], iterations=r3m["iterations"],
                          sample=samp, inv_ii={p["i"]: p["inv_ii"] for p in P3}, cut_ii={p["i"]: p["cut_ii"] for p in P3},
                          certifiable={p["i"]: p["certifiable"] for p in P3},
                          t_block={p["i"]: (p["t_local"], p["t_coeff"], p["t_cut"]) for p in P3})
        log("C3 K=3 on %d poses (K=2 elsewhere): rho(A) = %.4f, %.1fs" % (len(samp), r3m["rho"], timing["C3_K3_sample_patches_s"]))
        del P3, allP, r3m
    for K in list(c3):
        if isinstance(K, int):
            c3[K].pop("patches_raw", None)
    keep_P2 = None

    # ---------------------------------------------------------------- Krylov lower bounds
    t0 = time.perf_counter()
    if cfg["krylov"] == "all":
        kry_blocks = np.arange(n)
    else:
        rng = np.random.default_rng(1)
        lm = np.flatnonzero(model.kinds == "landmark")
        kry_blocks = np.sort(np.concatenate([np.flatnonzero(model.kinds != "landmark"),
                                             rng.choice(lm, size=min(KRYLOV_LANDMARK_SAMPLE, len(lm)), replace=False)]))
    both = _parallel_krylov(model, list(kry_blocks), (3, 5), workers)
    kry = {3: {i: v[3][0] for i, v in both.items()}, 5: {i: v[5][0] for i, v in both.items()}}
    timing["krylov_s"] = time.perf_counter() - t0
    log("Krylov bounds on %d blocks, %.1fs" % (len(kry_blocks), timing["krylov_s"]))

    # ---------------------------------------------------------------- C4 and hubs
    c4res = {}
    for w in cfg["C4_w"]:
        t0 = time.perf_counter()
        forced = np.flatnonzero(model.kinds == "marker") if cfg["hubs"] == "joint_markers" else None
        part = hubs.Partition(model, w, forced_hubs=forced)
        sc = hubs.SupernodeChain(model, part)
        kap = sc.kappa()
        GT = [G[d] for d in sc.dofs]
        # hub-clamped oracle error e~_T
        et = _clamped_error(model, part, G)
        u_sn = np.array([np.linalg.norm(sc.CD[k].T @ et[sc.dofs[k]], axis=0) for k in range(sc.n)])
        err_cl = np.full_like(err, np.nan)
        for b in part.T:
            err_cl[b] = np.linalg.norm(met.C[b].T @ et[model.block_slice(b)], axis=0)
        wres = dict(w=w, n_super=sc.n, n_hubs=int(len(part.S)), hub_dof=int(model.dims[part.S].sum()), kappa=kap,
                    err_clamped=err_cl, u_sn=u_sn, K={}, groups=part.groups, S=part.S)
        psi_all = {}
        for K in cfg["C4_K"]:
            tK = time.perf_counter()
            r4 = hubs.c4(sc, K, GT)
            psi = sc.psibar_links(K)
            psi_all[K] = psi
            mu_hi_T = np.array([ch.lam_max_scaled(sc.Pd[I], r4["cut_II"][I]) if r4["cut_II"][I] is not None else np.inf
                                for I in range(sc.n)])
            per = np.full_like(err, np.nan)
            if r4["E"] is not None:
                for I, g in enumerate(part.groups):
                    per[g] = np.sqrt(kap[g])[:, None] * r4["E"][I][None, :]
            KR = dict(rho=r4["rho"], rho_upper=r4["rho_upper"], iterations=r4["iterations"], c=r4["E"], s=r4["s"], A=r4["A"], E_per=per,
                      coeff=r4["coeff"], coeff_bound=r4["coeff_bound"], coeff_rows=r4["coeff_rows"], coeff_cols=r4["coeff_cols"],
                      mu_hi_T=mu_hi_T, psi=psi, cut_II=r4["cut_II"], certifiable=r4["certifiable"])
            if part.S.size and cfg["hubs"] == "joint_markers":
                KR.update(_joint_hub(model, sysm, met, part, sc, psi, mu_hi_T, r4, G, Etrue, err, kap))
            elif part.S.size and cfg["hubs"] == "one_at_a_time":
                if K in c3 and c3[K]["E"] is not None:
                    KR.update(_one_at_a_time(model, met, part, sc, psi, mu_hi_T, r4, c3[K]["E"], G, Etrue, err, kap))
                else:
                    KR["hub_note"] = "C3 at K=%d has rho(A) >= 1: no hub-block certificate E_h" % K
            KR["time_s"] = time.perf_counter() - tK
            wres["K"][K] = KR
            log("C4 w=%d K=%d: rho = %.4f (upper %.4f), %.1fs %s" % (w, K, r4["rho"], r4["rho_upper"], KR["time_s"],
                                                                     "E_S ok" if "E_S" in KR else KR.get("hub_note", "")))
        if part.S.size and cfg["hubs"] == "one_at_a_time":
            wres["hub_diag"] = _hub_diagnostics(model, sysm, met, part, Sig, G, Etrue)
        wres["time_s"] = time.perf_counter() - t0
        c4res[w] = wres

    timing["total_s"] = time.perf_counter() - t_all
    res["timing"] = timing
    # global energy check (Proposition 5.7): 1- and 5-step Gauss bounds and the oracle g^T H^-1 g per iteration
    t0 = time.perf_counter()
    glob = []
    for j in range(G.shape[1]):
        lb = radii.global_lower_bounds(model.H, G[:, j], steps=(1, 5))
        glob.append((lb[1], lb[5], float(G[:, j] @ Etrue[:, j])))
    timing["global_s"] = time.perf_counter() - t0
    return dict(model=model, info=info, res=res, err=err, err_xdiff=err_xdiff, Etrue=Etrue, G=G, X=X, xbar=xbar, Sig=Sig,
                lam=lam, cols=cols, c3=c3, kry=kry, kry_blocks=kry_blocks, c4=c4res, met=met, sysm=sysm, iters=iters,
                pose_floor=pose_floor, onestep=onestep, glob=np.array(glob), cfg=cfg)


# ------------------------------------------------------------------------------------------ pieces
_KW = {}


def _kry_worker(args):
    blocks, ells = args
    from threadpoolctl import threadpool_limits
    m = _KW["model"]
    with threadpool_limits(1):
        return [(int(i), covbounds.krylov_lower_multi(m, int(i), ells)) for i in blocks]


def _parallel_krylov(model, blocks, ells, workers, chunk=16):
    """{block: {l: (bound, dim)}} for every block, l in ells (one nested basis per block)."""
    import multiprocessing as mp
    if workers <= 1:
        return {int(i): covbounds.krylov_lower_multi(model, int(i), ells) for i in blocks}
    _KW["model"] = model
    parts = [(blocks[s:s + chunk], ells) for s in range(0, len(blocks), chunk)]
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(_kry_worker, parts)
    _KW.clear()
    return {r[0]: r[1] for part in res for r in part}


def _clamped_error(model, part, G):
    """e~_T = H_TT^-1 g_T (hub-clamped problem), in global dof numbering (zeros on S)."""
    Td = model.dof_index(part.T)
    HTT, HTT_lo = model.sub(Td)
    sub = SPDSystem(HTT, model.dims[part.T], model.elim[part.T], refine=3, K_lo=HTT_lo)
    assert sub.ok
    out = np.zeros_like(G)
    out[Td] = sub.solve(G[Td])
    return out


def _joint_hub(model, sysm, met, part, sc, psi, mu_hi_T, r4, G, Etrue, err, kap):
    """ChArUco: S = all markers jointly, hub window = whole graph (Shat_loc = Shat, theta_S exact)."""
    S = part.S
    Sd = model.dof_index(S)
    Td = model.dof_index(part.T)
    H = model.H.tocsr()
    HSS = H[Sd][:, Sd].toarray()
    HTS = H[Td][:, Sd].toarray()
    HTT, HTT_lo = model.sub(Td)
    sub = SPDSystem(HTT, model.dims[part.T], model.elim[part.T], refine=3, K_lo=HTT_lo)
    Xs = sub.solve(HTS, R_lo=model.H_lo[Td][:, Sd].toarray())   # H_TT^-1 H_TS
    Shat = 0.5 * ((HSS - HTS.T @ Xs) + (HSS - HTS.T @ Xs).T)
    theta = hubs.theta(HSS, Shat)                   # window = whole graph: theta_S = theta_S oracle
    CS = np.linalg.cholesky(HSS)
    mSJ, _, _, _ = hubs.hub_coupling_norms(model, sc, part, S)
    gS = G[Sd]
    et = sub.solve(G[Td])
    rS = gS - HTS.T @ et
    eS = Etrue[Sd]
    gS_dual = np.linalg.norm(sla.solve_triangular(CS, gS, lower=True), axis=0)
    rS_dual = np.linalg.norm(sla.solve_triangular(CS, rS, lower=True), axis=0)
    eS_norm = np.linalg.norm(CS.T @ eS, axis=0)
    out = dict(theta_S=theta, theta_S_oracle=theta, hub_window="whole graph", HSS_dim=int(len(Sd)))
    if r4["E"] is None:
        out["hub_note"] = "C4 rho(A) >= 1"
        return out
    chat = r4["E"]
    E_S = theta * (gS_dual + (mSJ[:, None] * chat).sum(axis=0))
    pb, pbt = hubs.phibar(sc, psi, mu_hi_T, mSJ, K=None)
    # oracle phi_IS (supernode) and per-block phi_iS
    CSinvT = sla.solve_triangular(CS, np.eye(len(CS)), lower=True).T
    pos_T = -np.ones(model.N, int)
    pos_T[Td] = np.arange(len(Td))
    phi_sn = np.array([np.linalg.norm(sc.CD[I].T @ Xs[pos_T[sc.dofs[I]]] @ CSinvT, 2) for I in range(sc.n)])
    n = model.n_blocks
    E_hub = np.full_like(err, np.nan)
    chat_per = np.full_like(err, np.nan)
    hub_term = np.full_like(err, np.nan)
    hub_term_or = np.full_like(err, np.nan)
    phi_blk = np.full(n, np.nan)
    phibar_blk = np.full(n, np.nan)
    corr = sub.solve(HTS @ eS)  # H_TT^-1 H_TS e_S (exact hub contribution to e_T)
    for I, g in enumerate(part.groups):
        for b in g:
            sl = model.block_slice(b)
            k = np.sqrt(kap[b])
            chat_per[b] = k * chat[I]
            hub_term[b] = k * pb[I] * E_S
            E_hub[b] = chat_per[b] + hub_term[b]
            hub_term_or[b] = np.linalg.norm(met.C[b].T @ corr[pos_T[np.arange(sl.start, sl.stop)]], axis=0)
            phi_blk[b] = np.linalg.norm(met.C[b].T @ Xs[pos_T[np.arange(sl.start, sl.stop)]] @ CSinvT, 2)
            phibar_blk[b] = k * pb[I]
    # hub blocks: H_SS is block diagonal (markers are not directly coupled), so ||e_h||_{H_hh} <= ||e_S||_{H_SS}
    blockdiag = model.adj[S][:, S].nnz == 0  # no factor joins two hubs: H_SS is block diagonal
    for h in S:
        E_hub[h] = E_S if blockdiag else np.inf
    # Proposition 4.7 hub-aware upper bound (per block in T)
    mu_hub = np.full(n, np.nan)
    for I, g in enumerate(part.groups):
        cut = r4["cut_II"][I]
        if cut is None:
            continue
        for b in g:
            o, d = sc.loc[b], int(model.dims[b])
            muT = mu_pair(met.C[b], cut[o:o + d, o:o + d])[1]
            mu_hub[b] = muT + theta * kap[b] * pb[I] ** 2
    out.update(E_S=E_S, eS_norm=eS_norm, energy_aggregate=theta * rS_dual, gS_dual=gS_dual,
               sum_term=(mSJ[:, None] * chat).sum(axis=0), mSJ=mSJ, phibar_sn=pb, phi_sn_oracle=phi_sn,
               E_hub=E_hub, chat_per=chat_per, hub_term=hub_term, hub_term_oracle=hub_term_or,
               phi_blk=phi_blk, phibar_blk=phibar_blk, mu_hi_hubaware=mu_hub, HSS_block_diagonal=bool(blockdiag))
    return out


def _one_at_a_time(model, met, part, sc, psi, mu_hi_T, r4, E3, G, Etrue, err, kap):
    """kitti06-prefix: E_i = sqrt(kappa_i) (c^_I + sum_h phibar_Ih E_h), E_h = C3 certificate of hub h."""
    if r4["E"] is None:
        return dict(hub_note="C4 rho(A) >= 1")
    chat = r4["E"]
    S = part.S
    H = model.H.tocsr()
    tot = np.zeros_like(chat)
    phibar_h = np.zeros((len(S), sc.n))
    for k, h in enumerate(S):
        hd = model.dof_index([h])
        Ch = np.linalg.cholesky(H[hd][:, hd].toarray())
        m = np.zeros(sc.n)
        for J in range(sc.n):
            Xj = H[hd][:, sc.dofs[J]]
            if Xj.nnz == 0:
                continue
            Y = sla.solve_triangular(Ch, Xj.toarray(), lower=True)
            Y = sla.solve_triangular(sc.CD[J], Y.T, lower=True).T
            m[J] = float(np.linalg.norm(Y, 2))
        pb, _ = hubs.phibar(sc, psi, mu_hi_T, m, K=None)
        phibar_h[k] = pb
        tot += pb[:, None] * E3[h][None, :]
    E_hub = np.full_like(err, np.nan)
    chat_per = np.full_like(err, np.nan)
    hub_term = np.full_like(err, np.nan)
    for I, g in enumerate(part.groups):
        kk = np.sqrt(kap[g])[:, None]
        chat_per[g] = kk * chat[I][None, :]
        hub_term[g] = kk * tot[I][None, :]
        E_hub[g] = chat_per[g] + hub_term[g]
    E_hub[S] = E3[S]
    # oracle hub contribution to e_T: e~_T - e_T = H_TT^-1 H_TS e_S
    Td = model.dof_index(part.T)
    Sd = model.dof_index(S)
    HTT, HTT_lo = model.sub(Td)
    sub = SPDSystem(HTT, model.dims[part.T], model.elim[part.T], refine=3, K_lo=HTT_lo)
    corr = sub.solve(H[Td][:, Sd] @ Etrue[Sd])
    pos_T = -np.ones(model.N, int)
    pos_T[Td] = np.arange(len(Td))
    hub_term_or = np.full_like(err, np.nan)
    for b in part.T:
        sl = model.block_slice(b)
        hub_term_or[b] = np.linalg.norm(met.C[b].T @ corr[pos_T[np.arange(sl.start, sl.stop)]], axis=0)
    return dict(E_hub=E_hub, chat_per=chat_per, hub_term=hub_term, hub_term_oracle=hub_term_or,
                phibar_hub_sn=phibar_h, hub_sum_sn=tot)


def _hub_diagnostics(model, sysm, met, part, Sig, G, Etrue):
    """Prefix: per-hub theta_h with the local window (Lemma 4.3: the hub, every pose from its first to its
    last observer, the factors among them) and the oracle theta_h (exact single-hub marginal precision);
    joint oracle theta_S and the energy aggregate theta_S ||r_S||_{H_SS^-1} for comparison."""
    from certify.check import sub_matrix
    H = model.H.tocsr()
    S = part.S
    rank = part.rank
    poses = np.flatnonzero(model.kinds == "pose")
    pose_by_rank = poses[np.argsort(rank[poses])]
    th_loc, th_or, spd = [], [], []
    for h in S:
        obs = [j for j in model.neighbours(h) if rank[j] >= 0]
        r0, r1 = min(rank[obs]), max(rank[obs])
        blocks = np.concatenate([pose_by_rank[r0:r1 + 1], [h]])
        Hp = sub_matrix(model, blocks)
        d = int(model.dims[h])
        A, Sd = slice(0, len(Hp) - d), slice(len(Hp) - d, len(Hp))
        try:
            L = np.linalg.cholesky(Hp[A, A])
            W = sla.solve_triangular(L, Hp[A, Sd], lower=True)
            Sl = Hp[Sd, Sd] - W.T @ W
            np.linalg.cholesky(Sl)
            Hhh = H[model.dof_index([h])][:, model.dof_index([h])].toarray()
            th_loc.append(hubs.theta(Hhh, Sl))
            spd.append(True)
        except np.linalg.LinAlgError:
            th_loc.append(np.inf)
            spd.append(False)
        Hhh = H[model.dof_index([h])][:, model.dof_index([h])].toarray()
        th_or.append(float(sla.eigh(Hhh, np.linalg.inv(Sig[h]), eigvals_only=True)[-1]))
    th_loc, th_or = np.array(th_loc), np.array(th_or)
    # joint oracle: Sigma_SS (all hubs jointly), theta_S^oracle and theta_S ||r_S||_{H_SS^-1} with r_S = Shat e_S
    Sd = model.dof_index(S)
    ES = np.zeros((model.N, len(Sd)))
    ES[Sd, np.arange(len(Sd))] = 1.0
    SigSS = sysm.solve(ES, refine=1, extended=False)[Sd]
    SigSS = 0.5 * (SigSS + SigSS.T)
    HSS = H[Sd][:, Sd].toarray()
    CS = np.linalg.cholesky(HSS)
    theta_joint = float(np.linalg.eigvalsh(CS.T @ SigSS @ CS)[-1])
    eS = Etrue[Sd]
    rS = sla.cho_solve(sla.cho_factor(SigSS, lower=True), eS)  # Shat e_S
    rS_dual = np.linalg.norm(sla.solve_triangular(CS, rS, lower=True), axis=0)
    eS_norm = np.linalg.norm(CS.T @ eS, axis=0)
    return dict(n_hubs=int(len(S)), hub_dof=int(len(Sd)), theta_local=th_loc, theta_oracle=th_or, window_spd=np.array(spd),
                window="poses from first to last observer (+ GP, se_unary factors on them, the hub's observations)",
                theta_S_joint_oracle=theta_joint, eS_norm_joint=eS_norm, energy_aggregate_joint_oracle=theta_joint * rS_dual)
