"""Stage 2b, Step 2 (BRIEF_stage2b.md §2): Theorem T1 at a certified stop, box options A and B (certify/STAGE2B.md).

Full D2 model in the chart at the stop state x_c (certify/t1.py). Blocks: poses (metric D_i = S_ii of the stop's
linearization, the certificate's metric) and kept landmarks (D_l = the landmark diagonal block of the Gauss-Newton matrix
at x_c).
  candidate z1   the Stage 2a global candidate (patch steps; landmark steps clipped to 0.1 m), in the x_c chart
  H0             F'(z1), the exact derivative (with the second-order terms), assembled in double
  pose row i     patch Q_i = P_i (the Stage 1 final patch) + every landmark P_i observes; landmark row l: patch {l}
                 s, a (leaks to exterior poses), l (patch weights) from H0 by exact block elimination of the landmarks
  Delta_kq       sup over the box of ||D_k^-1/2 (F'(zeta) - H0)_kq D_q^-1/2||, from the outward-rounded interval
                 enclosures of every factor's F'_f over z1 + B(u) (certify_stereo --t1-export), summed per block pair
  Psi(u)         (I - A)^-1 (s + l o rho(u)), rho_k(u) = sum_q Delta_kq u_q; check Psi(u) <= u, iterate to u+
  radii          poses c E_i (Stage 1 E at the stop); landmarks option A c 0.1 sqrt(lambda_max(D_l)), option B c E_l,
                 E_l = ||D_l^-1/2 g_l|| + sum_p ||D_l^-1/2 H_lp D_p^-1/2|| E_p (the stop's linearization); c in 1..16
The check is evaluated in double precision with rigorous enclosures for Delta. A certificate would additionally need the
verified linear algebra for s, a, l (certify/STAGE2B.md); it is only claimed if the check verifies.

Writes results/stage2b/<dataset>/<option>/probe_T1.csv and boxes.json, and the Delta statistics to intervals_<dataset>.json.

Usage: python3 scripts/stage2b_t1.py <dataset>
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from certify.io import Export  # noqa: E402
from certify.nonlinear_probe import FullModel  # noqa: E402
from certify.sidecar import Engine, Ops  # noqa: E402
from certify.sidecar_model import PoseModelBuilder, _landmark_init  # noqa: E402
from certify.t1 import T1Client, Problem  # noqa: E402

ETA, P_L, DOF = 0.1, 1e-6, 12
CS = (1, 2, 4, 8, 16)


def stage2a_candidate(ds):
    """The Stage 2a global candidate's step per export dof (patch steps; landmark steps clipped to 0.1 m)."""
    run = REPO / "exports/stage1" / ds / "corrected_stage2a" / "eval"
    ex = Export(run / "certified_stop_model")
    stop = json.load(open(run / "certified_stop_patches.json"))
    g2o = str(REPO / ex.meta["g2o"])
    m = PoseModelBuilder(p_L=P_L, g2o=g2o).build(None, ex=ex)
    S, x, b = m["S"], m["x"], m["b"]
    g = S @ x - b
    ids = [str(p) for p in m["pose_ids"]]
    n = len(ids)
    index = {p: k for k, p in enumerate(ids)}
    C = [np.linalg.cholesky(0.5 * (S[DOF * i:DOF * i + DOF, DOF * i:DOF * i + DOF] + S[DOF * i:DOF * i + DOF, DOF * i:DOF * i + DOF].T))
         for i in range(n)]
    Cinv = [sla.solve_triangular(c, np.eye(DOF), lower=True) for c in C]
    adj = np.abs(S.reshape(n, DOF, n, DOF)).max(axis=(1, 3)) > 0
    np.fill_diagonal(adj, False)
    eng = Engine(np.inf, ETA)
    eng._fcache, eng._fcacheF = {}, {}
    ops = Ops()
    rows = {i: eng._row(S, g, C, Cinv, adj, i, np.array(sorted(index[q] for q in stop["patches"][ids[i]]["P"])), ops) for i in range(n)}
    fm = FullModel(ex, p_L=P_L, g2o=g2o, init=_landmark_init(g2o))
    step = np.zeros(ex.n_dof)
    dP = {i: -sla.cho_solve(eng._fcache[rows[i].P.tobytes()], g[rows[i].dofP], check_finite=False) for i in range(n)}
    for i in range(n):
        k = int(np.flatnonzero(rows[i].P == i)[0])
        blk = fm.poses[i]
        step[ex.offsets[blk]:ex.offsets[blk] + DOF] = dP[i][DOF * k:DOF * k + DOF]
    observers = {}
    for i in range(n):
        for l in fm.adj[i]:
            observers.setdefault(int(l), []).append(i)
    unclipped = {}
    for l, obs in observers.items():
        q = max(obs, key=lambda q: (len(rows[q].P), -q))
        pd = (ex.offsets[fm.poses[rows[q].P]][:, None] + np.arange(DOF)[None, :]).ravel()
        sl = np.arange(ex.offsets[l], ex.offsets[l] + ex.dims[l])
        v = -fm.einv[l] @ (fm.g[sl] + fm.H[sl][:, pd] @ dP[q])
        unclipped[l] = float(np.linalg.norm(v[:3]))
        nv = unclipped[l]
        step[sl] = v if nv <= 0.1 else v * (0.1 / nv)
    return ex, fm, stop, ids, C, step, unclipped


def main(ds):
    t0 = time.perf_counter()
    exm, fm_lin, stop, ids, Cp, step, unclipped = stage2a_candidate(ds)
    ed = REPO / "exports/stage1" / ds / "corrected_stage2a" / "eval" / "certified_stop_beliefs"
    pb = Problem(ed)
    ex = pb.ex
    assert [str(p) for p in ex.var_id[pb.fm.poses]] == ids
    npose, nb = pb.npose, len(pb.blocks)
    z1 = np.zeros(pb.n)
    for k, b in enumerate(pb.blocks):
        z1[pb.boff[k]:pb.boff[k + 1]] = step[ex.offsets[b]:ex.offsets[b] + ex.dims[b]]
    # metrics: poses S_ii of the stop's linearization; landmarks the GN diagonal block at x_c
    H_gn = pb.fm.H
    C = list(Cp)
    for k in range(npose, nb):
        b = pb.blocks[k]
        sl = slice(ex.offsets[b], ex.offsets[b] + ex.dims[b])
        Dl = H_gn[sl][:, sl].toarray()
        C.append(np.linalg.cholesky(0.5 * (Dl + Dl.T)))
    Cinv = [sla.solve_triangular(c, np.eye(c.shape[0]), lower=True) for c in C]
    E_pose = np.array([stop["patches"][p]["E"] for p in ids])
    S_lo = np.array([stop["patches"][p]["S_lo"] for p in ids])

    cl = T1Client(ed, f"/dev/shm/gbp_os717_t1_{ds}", true_jacobians=True)
    F1, H0, fl1 = pb.full(z1, cl)
    H0 = H0.tocsr()
    blk = lambda M, k, q: M[pb.boff[k]:pb.boff[k + 1]][:, pb.boff[q]:pb.boff[q + 1]]
    npd = int(pb.boff[npose])

    # pose-landmark adjacency in the model
    Hpl = H0[:npd][:, npd:].tocsr()
    lm_of_pose = [np.unique(np.flatnonzero(np.abs(Hpl[DOF * i:DOF * i + DOF]).sum(axis=0).A1 > 0) // 3) + npose for i in range(npose)]
    assert all((pb.boff[npose + 1:] - pb.boff[npose:-1]) == 3), "landmark blocks are 3-dimensional"
    Hll_inv = {k: np.linalg.inv(blk(H0, k, k).toarray()) for k in range(npose, nb)}

    # rows: poses (patches Q_i = P_i + their landmarks), landmarks (trivial patches)
    pose_rows = {}
    cache = {}
    for i in range(npose):
        P = np.array(sorted(ids.index(q) for q in stop["patches"][ids[i]]["P"]))
        key = P.tobytes()
        if key not in cache:
            L = np.unique(np.concatenate([lm_of_pose[p] for p in P]))
            pd = np.concatenate([np.arange(pb.boff[p], pb.boff[p + 1]) for p in P])
            ld = np.concatenate([np.arange(pb.boff[l], pb.boff[l + 1]) for l in L])
            HPL = H0[pd][:, ld].toarray()
            HLP = H0[ld][:, pd].toarray()
            Linv = sla.block_diag(*[Hll_inv[l] for l in L])
            Sq = H0[pd][:, pd].toarray() - HPL @ Linv @ HLP
            lu = sla.lu_factor(Sq)
            # exterior poses: adjacent to P via GP (pose-pose) or to L via observations
            inP = np.zeros(npose, bool)
            inP[P] = True
            ext = set()
            for p in P:
                for q in range(npose):
                    if not inP[q] and blk(H0, p, q).nnz:
                        ext.add(q)
            for l in L:
                for q in np.flatnonzero(np.abs(H0[pb.boff[l]:pb.boff[l + 1]][:, :npd]).sum(axis=0).A1 > 0) // DOF:
                    if not inP[q]:
                        ext.add(int(q))
            cache[key] = dict(P=P, L=L, pd=pd, ld=ld, HPL=HPL, Linv=Linv, lu=lu, ext=sorted(ext))
        c = cache[key]
        k = int(np.flatnonzero(c["P"] == i)[0])
        # row i of Sq^-1: solve Sq' y = e_i (12 columns)
        Ei = np.zeros((len(c["pd"]), DOF))
        Ei[DOF * k:DOF * k + DOF] = np.eye(DOF)
        Y = sla.lu_solve(c["lu"], Ei, trans=1)  # Sq^-T E_i ; row i of Sq^-1 = Y'
        Rrow = C[i].T @ Y.T  # C_i' [Sq^-1]_{i,:}
        Fq = F1[c["pd"]] - c["HPL"] @ (c["Linv"] @ F1[c["ld"]])
        s = float(np.linalg.norm(Rrow @ Fq))
        a = {}
        for j in c["ext"]:
            jd = np.arange(pb.boff[j], pb.boff[j + 1])
            M = H0[c["pd"]][:, jd].toarray() - c["HPL"] @ (c["Linv"] @ H0[c["ld"]][:, jd].toarray())
            a[j] = float(np.linalg.norm(Rrow @ M @ Cinv[j].T, 2))
        ell = {}
        for m, p in enumerate(c["P"]):
            ell[int(p)] = float(np.linalg.norm(Rrow[:, DOF * m:DOF * m + DOF] @ C[p], 2))
        W = -(Rrow @ c["HPL"])  # C_i' [Sq^-1 (-H_PL)]_{i,:}, landmark columns (times H_ll^-1 below)
        for m, l in enumerate(c["L"]):
            ell[int(l)] = float(np.linalg.norm(W[:, 3 * m:3 * m + 3] @ Hll_inv[l] @ C[l], 2))
        pose_rows[i] = dict(s=s, a=a, ell=ell, patch=len(c["P"]), landmarks=len(c["L"]))
    lm_rows = {}
    for l in range(npose, nb):
        ld = np.arange(pb.boff[l], pb.boff[l + 1])
        R = C[l].T @ Hll_inv[l]
        obs = np.flatnonzero(np.abs(H0[ld][:, :npd]).sum(axis=0).A1 > 0) // DOF
        lm_rows[l] = dict(s=float(np.linalg.norm(R @ F1[ld])),
                          a={int(j): float(np.linalg.norm(R @ H0[ld][:, pb.boff[j]:pb.boff[j + 1]].toarray() @ Cinv[j].T, 2)) for j in np.unique(obs)},
                          ell={l: float(np.linalg.norm(R @ C[l], 2))})
    # A over poses (landmark rows have no leak to landmarks, pose rows none to landmarks)
    App = np.zeros((npose, npose))
    for i, r in pose_rows.items():
        for j, v in r["a"].items():
            App[i, j] = v
    rho_row = App.sum(axis=1).max()
    w = np.ones(npose)
    for _ in range(200):
        w = App @ w + 1e-3
        w /= w.max()
    theta = float(np.max((App @ w) / w))
    cw = dict(max_row_sum=float(rho_row), perron_witness_ratio=theta, certified_rho_lt_1=bool(theta < 1.0))

    # landmark radii
    lam_max = np.array([np.linalg.eigvalsh(C[l] @ C[l].T)[-1] for l in range(npose, nb)])
    uA = 0.1 * np.sqrt(lam_max)
    g_lin = fm_lin.g
    H_lin = fm_lin.H
    E_l = np.zeros(nb - npose)
    for k, l in enumerate(range(npose, nb)):
        b = pb.blocks[l]
        sl = slice(ex.offsets[b], ex.offsets[b] + ex.dims[b])
        val = float(np.linalg.norm(Cinv[l] @ g_lin[sl]))
        for j in lm_rows[l]["a"]:
            bj = pb.blocks[j]
            val += float(np.linalg.norm(Cinv[l] @ H_lin[sl][:, ex.offsets[bj]:ex.offsets[bj] + DOF].toarray() @ Cinv[j].T, 2)) * E_pose[j]
        E_l[k] = val

    s_vec = np.array([pose_rows[i]["s"] for i in range(npose)] + [lm_rows[l]["s"] for l in range(npose, nb)])
    results, deltas = {}, {}
    for opt in ("A", "B"):
        res_c = []
        for c in CS:
            u_bar = np.concatenate([c * E_pose, c * (uA if opt == "A" else E_l)])
            # coordinate radii enclosing the D-norm box: |w_k,j| <= u_k sqrt([D_k^-1]_jj) (inflated by 1e-10)
            rad = np.zeros(pb.n)
            for k in range(nb):
                Dinv_diag = np.sum(Cinv[k] ** 2, axis=0)  # diag(C^-T C^-1) = diag(D^-1)
                rad[pb.boff[k]:pb.boff[k + 1]] = u_bar[k] * np.sqrt(Dinv_diag) * (1 + 1e-10)
            enc = cl.enclose(pb.export_z(z1), pb.export_z(rad))
            # Delta per block pair from the enclosures (summed over factors), minus H0
            Dlt = {}
            unbounded = []
            for t in ("stereo", "gp"):
                f = pb.fac[t]
                kk = np.flatnonzero(f["keep"])
                V = f["vars"][kk]
                lo, hi, flg = enc[t]["dFlo"][kk], enc[t]["dFhi"][kk], enc[t]["flags"][kk]
                dims_f = [int(ex.dims[V[0, s_]]) for s_ in range(V.shape[1])]
                offs = np.concatenate([[0], np.cumsum(dims_f)])
                for r_ in range(len(kk)):
                    bl = [int(pb.pos[v]) for v in V[r_]]
                    bad = bool(flg[r_]) or not np.all(np.isfinite(lo[r_])) or not np.all(np.isfinite(hi[r_]))
                    if bad:  # unbounded enclosure, or G-solver's F not smooth on the box: Delta = inf for its blocks
                        unbounded.append(dict(type=t, factor=str(ex.factors[t]["ids"][kk[r_]]), flags=int(flg[r_]), blocks=bl))
                    for a_ in range(len(bl)):
                        for b_ in range(len(bl)):
                            key = (bl[a_], bl[b_])
                            lo_ab = lo[r_][offs[a_]:offs[a_ + 1], offs[b_]:offs[b_ + 1]]
                            hi_ab = hi[r_][offs[a_]:offs[a_ + 1], offs[b_]:offs[b_ + 1]]
                            if bad:
                                lo_ab = np.full_like(lo_ab, -np.inf)
                                hi_ab = np.full_like(hi_ab, np.inf)
                            if key in Dlt:
                                Dlt[key][0] += lo_ab
                                Dlt[key][1] += hi_ab
                            else:
                                Dlt[key] = [lo_ab.copy(), hi_ab.copy()]
            # pseudo-factors are exact in H0 (linear): no contribution
            Delta = {}
            for (k, q), (lo_, hi_) in Dlt.items():
                H0b = blk(H0, k, q).toarray()
                mag = np.maximum(np.abs(lo_ - H0b), np.abs(hi_ - H0b))
                if not np.all(np.isfinite(mag)):
                    Delta[(k, q)] = np.inf
                else:
                    Delta[(k, q)] = float(np.linalg.norm(np.abs(Cinv[k]) @ mag @ np.abs(Cinv[q]).T, "fro") * (1 + 1e-12))

            def rho(u):
                r = np.zeros(nb)
                for (k, q), d in Delta.items():
                    r[k] += d * u[q] if np.isfinite(d) else (np.inf if u[q] > 0 else 0.0)
                return r

            def Psi(u):
                rr = rho(u)
                v = s_vec.copy()
                for i in range(npose):
                    v[i] += sum(e * rr[p] for p, e in pose_rows[i]["ell"].items())
                for l in range(npose, nb):
                    v[l] += lm_rows[l]["ell"][l] * rr[l]
                if not np.all(np.isfinite(v[:npose])):
                    uP = np.where(np.isfinite(v[:npose]), 0.0, np.inf)
                    uP = np.full(npose, np.inf) if not np.all(np.isfinite(v[:npose])) else uP
                else:
                    uP = np.linalg.solve(np.eye(npose) - App, v[:npose])
                uL = v[npose:].copy()
                for k, l in enumerate(range(npose, nb)):
                    uL[k] += sum(a * uP[j] for j, a in lm_rows[l]["a"].items())
                return np.concatenate([uP, uL])

            ps = Psi(u_bar)
            ok = bool(np.all(ps <= u_bar))
            ratio = ps / u_bar
            rec = dict(c=c, verified=ok, poses_ok=int(np.sum(ps[:npose] <= u_bar[:npose])), landmarks_ok=int(np.sum(ps[npose:] <= u_bar[npose:])),
                       landmarks=nb - npose, pose_ratio_max=float(np.max(ratio[:npose])), pose_ratio_median=float(np.median(ratio[:npose])),
                       landmark_ratio_max=float(np.max(ratio[npose:])), landmark_ratio_p99=float(np.percentile(ratio[npose:], 99)),
                       landmarks_failing=int(np.sum(ps[npose:] > u_bar[npose:])), unbounded_factors=len(unbounded),
                       unbounded_types={t: sum(1 for x in unbounded if x["type"] == t) for t in ("stereo", "gp")},
                       poses_touching_unbounded=len({b for x in unbounded for b in x["blocks"] if b < npose}),
                       s_landmark_over_u_max=float(np.max(s_vec[npose:] / u_bar[npose:])),
                       s_pose_over_u=_q((s_vec[:npose] / u_bar[:npose]).tolist()),
                       poses_with_infinite_Psi=int(np.sum(~np.isfinite(ps[:npose]))),
                       landmarks_s_over_u=int(np.sum(s_vec[npose:] > u_bar[npose:])),
                       delta_stats={"pose-pose": _q([d for (k, q), d in Delta.items() if k < npose and q < npose]),
                                    "pose-landmark": _q([d for (k, q), d in Delta.items() if (k < npose) != (q < npose)]),
                                    "landmark-landmark": _q([d for (k, q), d in Delta.items() if k >= npose and q >= npose])})
            if ok:  # iterate to u+ (the certificate would still need the verified linear algebra of certify/STAGE2B.md)
                u = u_bar
                for _ in range(200):
                    un = Psi(u)
                    if np.max(np.abs(un - u) / np.maximum(u, 1e-300)) < 1e-12:
                        u = un
                        break
                    u = un
                rec["u_plus"] = u[:npose].tolist()
            res_c.append(rec)
            deltas[(opt, c)] = rec["delta_stats"]
            print("%s option %s c=%d: verified %s | poses ok %d/%d (max Psi/u %.3g) | landmarks failing %d/%d (s_l > u_l: %d; max Psi/u %.3g) | unbounded factors %d %s" % (
                ds, opt, c, ok, rec["poses_ok"], npose, rec["pose_ratio_max"], rec["landmarks_failing"], nb - npose,
                rec["landmarks_s_over_u"], rec["landmark_ratio_max"], len(unbounded), rec["unbounded_types"]), flush=True)
            if ok:
                break
        results[opt] = res_c
    cl.close()

    # per pose: z1, s, and (only if verified) u+, E_NL; the true distance from the Newton reference
    zstar = np.load(REPO / f"exports/stage2b/{ds}/z_star.npy")
    out = REPO / "results/stage2b" / ds
    for opt in ("A", "B"):
        (out / opt).mkdir(parents=True, exist_ok=True)
        ver = [r for r in results[opt] if r["verified"]]
        with open(out / opt / "probe_T1.csv", "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["pose", "z1_norm", "s", "u_plus", "E_NL", "lower_bound", "true_distance", "eta_S_lo", "E_NL_le_eta_S_lo", "certified", "why"])
            for i, p in enumerate(ids):
                z1n = float(np.linalg.norm(C[i].T @ z1[pb.boff[i]:pb.boff[i + 1]]))
                td = float(np.linalg.norm(C[i].T @ zstar[pb.boff[i]:pb.boff[i + 1]]))
                if ver:
                    up = ver[0]["u_plus"][i]
                    wr.writerow([p, z1n, pose_rows[i]["s"], up, z1n + up, max(0.0, z1n - up), td, ETA * S_lo[i], int(z1n + up <= ETA * S_lo[i]), 1, ""])
                else:
                    wr.writerow([p, z1n, pose_rows[i]["s"], "", "", "", td, ETA * S_lo[i], "", 0, "Psi(u) <= u not verified for c in 1..16"])
        (out / opt / "boxes.json").write_text(json.dumps(dict(option=opt, collatz_wielandt=cw, attempts=results[opt],
                                                               landmark_radius=dict(option_A_median=float(np.median(uA)), option_B_median=float(np.median(E_l)),
                                                                                    option_B_max=float(np.max(E_l)))), indent=1, default=float))
    summ = dict(dataset=ds, poses=npose, landmarks=nb - npose, H0="exact derivative F'(z1) (second-order terms included)",
                s_pose=_q([pose_rows[i]["s"] for i in range(npose)]), s_landmark=_q(s_vec[npose:].tolist()),
                s_landmark_over_uA1=_q((s_vec[npose:] / uA).tolist()), s_landmark_over_uB1=_q((s_vec[npose:] / np.maximum(E_l, 1e-300)).tolist()),
                unclipped_landmark_steps=_q(list(unclipped.values())), flags_at_z1=int(fl1), collatz_wielandt=cw,
                delta={"%s_c%d" % k: v for k, v in deltas.items()}, seconds=time.perf_counter() - t0)
    (out / "t1_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps({k: v for k, v in summ.items() if k != "delta"}, indent=1, default=float))


def _q(a):
    a = np.asarray([x for x in a if np.isfinite(x)], float)
    if not len(a):
        return dict(n=0)
    return dict(n=int(len(a)), median=float(np.median(a)), p95=float(np.percentile(a, 95)), max=float(np.max(a)))


if __name__ == "__main__":
    main(sys.argv[1])
