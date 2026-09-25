# Proof Note: Self-Certifying GBP for Continuous-Time SLAM

> **Errata, 2026-09-25** (from Phase A, facts F3–F5; theorems unchanged)
>
> (i) **F4.** On the chain, $G_k=(H_{kk}-\Lambda^f_{kk})+\text{message}_{k+1\to k}$, where $\Lambda^f_{kk}$ is the forward GP factor's diagonal contribution to $H_{kk}$ and $\text{message}_{k+1\to k}$ is the stored factor-to-variable message of the GP factor $f(k,k+1)$ to $k$. The remark "$G_k=P_{kk}+$ message" in proof note §1.1 and brief §3 is wrong by that term. Lemma 1.4's seed $P_{mm}^-$ is exactly $H_{mm}-\Lambda^f_{mm}$, consistent with this. Nothing else changes. (Verified to relative error $10^{-11}$ on a 600-pose GP chain; `results/phaseA/f4_chain.json`.)
>
> (ii) **F3.** G-solver's interpolation is affine in the GP local variables, hence in tangent coordinates within a fixed chart only to first order. Theorem 5.6 applies to the linearized interpolant in the frozen charts. Phase E must report the interpolant's linearization error at observed tangent magnitudes as a caveat.
>
> (iii) **F5.** There is a factor of 2 in every stereo rotation Jacobian (`se3_posevel_stereo_point.cpp:77`). It is recorded and left unchanged: the certified model is the assembled one.

Version of 2026-09-24. Every result is stated against five checkable properties of G-solver (P1–P5), never against G-solver itself. No proof depends on a measurement; the measurements only select which theorem applies. Status tags: **[classical]** cited, not claimed; **[verified]** classical or elementary, checked line by line; **[new]** proved here; **[conditional]** proved here under a stated fact about G-solver still to be read off the code.

---

## 0. Conventions and standing assumptions

### 0.1 The linear model

At any instant every factor $f$ has a linearization point, a whitened Jacobian $J_f$ and whitened residual $r_f$. The **current model** is
$$H=\sum_f J_f^\top\Lambda_f J_f+G,\qquad b=\sum_f J_f^\top\Lambda_f r_f+b_G,$$
where $G\succeq 0$ is the gauge-fixing (anchor/prior) term. This covers synchronous and asynchronous relinearization alike: $H,b$ are well defined at every instant. The **exact solution of the current model** is $\bar x=H^{-1}b$. Nothing about the true state $x^\star$ enters until §5.

**(A0)** $H\succ 0$. How $G$ is chosen is property P5.

### 0.2 Blocks, graph, notation

Variables are blocks $i\in V$ of dimension $d_i$. $H_{ij}$ is the $(i,j)$ block. The **variable graph** joins $i,j$ when a factor touches both; $d_G$ is its distance. $D=\mathrm{blkdiag}(H_{11},\dots,H_{nn})$, $D_i=H_{ii}$. $\|v\|_W=(v^\top Wv)^{1/2}$ for SPD $W$; $\|X\|$ is the spectral norm; $\preceq$ is the Loewner order. $\Sigma=H^{-1}$, $\Sigma_{ii}=[H^{-1}]_{ii}$.

For any estimate $x$: **residual** $g=Hx-b$, **error** $e=x-\bar x$, so $He=g$ exactly. $g_i=\sum_j H_{ij}x_j-b_i$ is computable at node $i$ from its neighbours.

### 0.3 Metric

Errors of block $i$ are measured in $\|\cdot\|_{D_i}$. It is unit-free (whitens the mixed coordinates of a 12-dimensional node) and makes the statistical yardstick free: $\Sigma_{ii}\succeq H_{ii}^{-1}$ (Lemma 2.1, trivial subspace), so the $(1-\delta)$ radius of the exact estimator's error in this metric is at least $\sqrt{\chi^2_{d_i,1-\delta}}$ with no computation (Proposition 2.7). Conversion to another SPD metric $W_i$ is by the factor $\|W_i^{1/2}D_i^{-1/2}\|$.

### 0.4 The five properties

| | Question | Determined by |
|---|---|---|
| **P1** chain coupling | Does the classical $\rho(M)$ approach 1 as the trajectory grows? | Measured (§3.6 predicts yes for prior-dominated segments) |
| **P2** hubs | Are there landmarks whose track spans more than $w$ consecutive poses? | Factor list; $w$ is a design choice (§4) |
| **P3** local information | After cutting a patch's boundary factors, is the patch nondegenerate at small $K$? | Measured (Definition 2.5) |
| **P4** relinearization | Synchronous or per-factor asynchronous? | Code; affects only §5.2 |
| **P5** gauge | Single anchor prior, or per-robot/distributed anchoring? | Code; affects (A0) and hub nondegeneracy |

---

## 1. Chain lemmas

Let $P$ be block-tridiagonal SPD with blocks $P_{kl}$, $P_{kl}=0$ for $|k-l|>1$, and $\Sigma=P^{-1}$. In §4, $P$ is $H$ restricted to the non-hub variables in supernodes.

### 1.1 Backward and forward Schur complements

**Definition 1.1.** $G_n=P_{nn}$, $G_k=P_{kk}-P_{k,k+1}G_{k+1}^{-1}P_{k+1,k}$ for $k=n-1,\dots,1$; forward $F_1=P_{11}$, $F_k=P_{kk}-P_{k,k-1}F_{k-1}^{-1}P_{k-1,k}$. All SPD. $G_k$ is the precision of $x_k$ in the sub-chain $\{k,\dots,n\}$ with $x_{k-1}$ clamped; on a GBP chain, $G_k=P_{kk}+$ (converged message from $k+1$).

### 1.2 Transfer operators and the product formula

**Lemma 1.2 [verified; classical, e.g. Meurant 1992].** With $\Psi_k=-G_{k+1}^{-1}P_{k+1,k}$, for $i<j$,
$$\Sigma_{ij}=\Sigma_{ii}\,\Psi_i^\top\Psi_{i+1}^\top\cdots\Psi_{j-1}^\top,$$
and symmetrically for $i>j$ with $\Phi_k=-F_{k-1}^{-1}P_{k-1,k}$.

*Proof.* Let $x\sim\mathcal N(0,\Sigma)$. Block-tridiagonal precision makes $x$ a Markov chain: $x_j\mid x_{1:j-1}$ depends only on $x_{j-1}$. Conditioning on $x_{1:j-1}$, the block $x_{j:n}$ has precision $P_{[j:n],[j:n]}$ and mean $m$ solving $P_{[j:n],[j:n]}m=-e_jP_{j,j-1}x_{j-1}$. Hence $m_j=-[P_{[j:n],[j:n]}^{-1}]_{jj}P_{j,j-1}x_{j-1}=-G_j^{-1}P_{j,j-1}x_{j-1}=\Psi_{j-1}x_{j-1}$, since the $(j,j)$ block of that inverse is $G_j^{-1}$. So $\mathrm{Cov}(x_i,x_j)=\mathrm{Cov}(x_i,\mathbb E[x_j\mid x_{1:j-1}])=\mathrm{Cov}(x_i,x_{j-1})\Psi_{j-1}^\top$. Induct from $j=i+1$. $\square$

Influence from $j$ to $i$ is a product of one operator per link, so decay is certified link by link, not by a global spectrum. This replaces Demko–Moss–Smith, whose constants would need $\lambda_{\min}$ of the whole matrix.

### 1.3 Local lower bounds on $G_k$

**Lemma 1.3 [new, elementary].** If $L_{k+1}\preceq G_{k+1}$, both SPD, then $L_k:=P_{kk}-P_{k,k+1}L_{k+1}^{-1}P_{k+1,k}\preceq G_k$.

*Proof.* $L_{k+1}^{-1}\succeq G_{k+1}^{-1}$, so the subtracted term is larger, so $L_k\preceq G_k$. $\square$

**Lemma 1.4 (truncation seed) [new, elementary].** Let $P_{mm}^{-}$ be $P_{mm}$ with the contribution of every factor touching a node beyond $m$ removed. Then $G_m\succeq P_{mm}^{-}$.

*Proof.* $G_m$ is the conditional precision of $x_m$ given $x_{m-1}$ after marginalizing the future. Deleting factors deletes PSD terms from the joint precision, enlarging the joint covariance, hence the conditional covariance of $x_m$ given $x_{m-1}$, hence lowering its conditional precision. In the deleted graph $x_m$ is disconnected from the future, so that precision is $P_{mm}^{-}$. $\square$

Starting from $L_m=P_{mm}^-$ at any truncation node and recursing backward gives valid SPD lower bounds on every $G_k$, $k<m$, using only blocks between $k$ and $m$. In G-solver the seed is SPD because the GP factor to the past contributes a full-rank $12\times12$ block.

### 1.4 Local decay factors

**Definition 1.5.** For link $(k,k+1)$ and $L_{k+1}\preceq G_{k+1}$,
$$\bar\psi_k=\lambda_{\max}\!\big(D_{k+1}^{1/2}L_{k+1}^{-1}D_{k+1}^{1/2}\big)\cdot\big\|D_{k+1}^{-1/2}P_{k+1,k}D_k^{-1/2}\big\|,\qquad D_k=P_{kk}.$$

**Lemma 1.6 (decay of influence) [new].** For $i<j$,
$$\big\|D_i^{1/2}\Sigma_{ij}D_j^{1/2}\big\|\le\lambda_{\max}\!\big(D_i^{1/2}\Sigma_{ii}D_i^{1/2}\big)\prod_{k=i}^{j-1}\bar\psi_k.$$

*Proof.* Insert $D^{1/2}D^{-1/2}$ between the factors of Lemma 1.2 and take norms. For each factor, $\|D_{k+1}^{1/2}\Psi_kD_k^{-1/2}\|\le\lambda_{\max}(D_{k+1}^{1/2}G_{k+1}^{-1}D_{k+1}^{1/2})\,\|D_{k+1}^{-1/2}P_{k+1,k}D_k^{-1/2}\|\le\bar\psi_k$, using $G_{k+1}^{-1}\preceq L_{k+1}^{-1}$. $\square$

The second factor of $\bar\psi_k$ is the normalized coupling ($\approx\tfrac12$ on a prior-dominated WNOA link); the first is $\ge1$ and measures how much of node $k+1$'s information survives marginalizing the future. On a pure prior chain $\bar\psi_k\approx1$: no decay, which is correct. If P3 fails on a segment, Lemma 1.6 certifies no decay across it and the window must extend to the next informative measurement.

---

## 2. Marginal covariance bounds and the statistical yardstick

### 2.1 The subspace lower bound

**Lemma 2.1 [verified; classical].** Let $V$ have orthonormal columns whose span contains the coordinate directions $E_i$ of block $i$. Then
$$\Sigma_{ii}=E_i^\top H^{-1}E_i\;\succeq\;E_i^\top V\,(V^\top HV)^{-1}V^\top E_i.$$

*Proof.* Complete to an orthogonal $[V,V_\perp]$. In that basis $[H^{-1}]_{VV}=(H_{VV}-H_{V\perp}H_{\perp\perp}^{-1}H_{\perp V})^{-1}\succeq H_{VV}^{-1}$ because the subtracted term is PSD and inversion is antitone. Since $E_i\in\mathrm{span}\,V$, $E_i^\top H^{-1}E_i=E_i^\top V[H^{-1}]_{VV}V^\top E_i\succeq E_i^\top VH_{VV}^{-1}V^\top E_i$. $\square$

Instances: **trivial** $V=E_i$: $\Sigma_{ii}\succeq H_{ii}^{-1}$. **Patch** $V$ = coordinates of a $K$-hop patch: $\Sigma_{ii}\succeq[(H_{PP})^{-1}]_{ii}$. **Krylov** $V=\mathrm{orth}\{E_i,HE_i,\dots,H^{\ell-1}E_i\}$: the block Gauss quadrature bound, each step one sparse product.

**Lemma 2.2 (monotonicity) [verified].** $\mathrm{span}V\subseteq\mathrm{span}V'$ implies the $V'$ bound dominates. *Proof.* Lemma 2.1 inside $V'$ with $H$ replaced by $V'^\top HV'$. $\square$

### 2.2 The boundary-cut upper bound

**Definition 2.3.** For patch $P$, let $F_\partial$ be the factors touching both $P$ and its exterior $E$, with $J_f=[J_{f,P},J_{f,E}]$. Set $B_P=\sum_{f\in F_\partial}J_{f,P}^\top\Lambda_fJ_{f,P}\succeq0$.

**Theorem 2.4 [new in this form; ingredients classical].** If $H_{PP}-B_P\succ0$ then
$$\Sigma_{ii}\;\preceq\;\big[(H_{PP}-B_P)^{-1}\big]_{ii}.$$

*Proof.* The marginal precision of $P$ is $S_P=H_{PP}-H_{PE}H_{EE}^{-1}H_{EP}$ with $\Sigma_{PP}=S_P^{-1}$. It suffices to show $H_{PE}H_{EE}^{-1}H_{EP}\preceq B_P$. Stack $U=[\Lambda_f^{1/2}J_{f,P}]_{f\in F_\partial}$, $W=[\Lambda_f^{1/2}J_{f,E}]_{f\in F_\partial}$; then $H_{PE}=U^\top W$, $B_P=U^\top U$, and $H_{EE}\succeq W^\top W$ (the other contributions are PSD), so $H_{EE}^{-1}\preceq(W^\top W)^{+}$ on range $W$. Then $H_{PE}H_{EE}^{-1}H_{EP}\preceq U^\top W(W^\top W)^{+}W^\top U=U^\top\Pi U\preceq U^\top U=B_P$ with $\Pi$ the orthogonal projector onto range $W$. Hence $S_P\succeq H_{PP}-B_P$; invert and take the $(i,i)$ block. $\square$

**Definition 2.5 (P3).** $P$ is **locally certifiable** if $H_{PP}-B_P\succ0$: the factors wholly inside $P$ (plus the gauge term if the anchor is in $P$) determine every variable in $P$. Otherwise the upper bound is $+\infty$, which is the honest verdict.

**Lemma 2.6 [verified].** $P\subset P'$ implies the $P'$ upper bound is at most the $P$ upper bound. *Proof.* The cut model on $P'$ contains every factor of the cut model on $P$ plus more PSD terms. $\square$

### 2.3 Statistical radii

Under the Gaussian noise model $\bar x_i-x_i^\star\sim\mathcal N(0,\Sigma_{ii})$. Let $q_d(\delta)=\sqrt{\chi^2_{d,1-\delta}}$.

**Proposition 2.7 [verified].** With $\mu_{\mathrm{lo}}\le\lambda_{\min}(D_i^{1/2}\Sigma_{ii}D_i^{1/2})$ and $\mu_{\mathrm{hi}}\ge\lambda_{\max}(\cdot)$, the $(1-\delta)$ quantile $r_i^\star(\delta)$ of $\|\bar x_i-x_i^\star\|_{D_i}$ satisfies
$$\underline S_i(\delta):=\sqrt{\mu_{\mathrm{lo}}}\,q_{d_i}(\delta)\le r_i^\star(\delta)\le\sqrt{\mu_{\mathrm{hi}}}\,q_{d_i}(\delta)=:\overline S_i(\delta).$$

*Proof.* With $Q=D_i^{1/2}\Sigma_{ii}D_i^{1/2}$ (eigenvalues $\mu_k$) and $z\sim\mathcal N(0,I)$, $\|\bar x_i-x_i^\star\|_{D_i}^2\sim\sum_k\mu_kz_k^2$, stochastically between $\mu_{\mathrm{lo}}\chi^2_d$ and $\mu_{\mathrm{hi}}\chi^2_d$. $\square$

**Corollary 2.8.** Trivial subspace gives $\mu_{\mathrm{lo}}=1$, so $\underline S_i=q_{d_i}(\delta)$ always. Patch/Krylov give $\mu_{\mathrm{lo}}\ge1$.

---

## 3. Computational certificates

### 3.1 Classical comparison bound

**Theorem 3.1 [classical: Malioutov–Johnson–Willsky 2006; Feingold–Varga 1962].** $B=D^{-1/2}HD^{-1/2}=I-R$, $R_{ii}=0$, $M_{ij}=\|R_{ij}\|$, $M_{ii}=0$. If $\rho(M)<1$, with $C=(I-M)^{-1}$ and $\nu_j=\|D_j^{-1/2}g_j\|$,
$$\|e_i\|_{D_i}\le[C\nu]_i,\qquad\|D_i^{1/2}\Sigma_{ij}D_j^{1/2}\|\le C_{ij}.$$
*Proof.* $B^{-1}=\sum_kR^k$ with $\|(R^k)_{ij}\|\le(M^k)_{ij}$ by submultiplicativity along paths. $\square$

### 3.2 Backbone-preconditioned comparison bound

Let $P$ be block-tridiagonal SPD: GP chain factors + block-diagonal measurement information + $G$; $R_H=H-P$ holds only off-diagonal measurement couplings. $\hat M_{ij}=\|[P^{-1}R_H]_{ij}\|$, $\hat M_{ii}=0$.

**Theorem 3.2 [new as stated].** If $\rho(\hat M)<1$, with $\hat C=(I-\hat M)^{-1}$ and $\hat\nu_j=\|[P^{-1}g]_j\|_{D_j}$,
$$\|e_i\|_{D_i}\le[\hat C\hat\nu]_i.$$
$P=D$ recovers Theorem 3.1.

*Proof.* $(I+P^{-1}R_H)e=P^{-1}g$, so $e=\sum_k(-P^{-1}R_H)^kP^{-1}g$, dominated blockwise by $\hat M^k$. $\square$

On a graph with no landmark couplings $R_H=0$ and the certificate is exact. $P$ must include measurement information: the pure GP chain is singular without an anchor and its anchored inverse grows like a random walk, so it would not decay.

### 3.3 The exact patch identity

**Lemma 3.3 [verified].** For patch $P$, exterior $E$, $i\in P$:
$$e_P=H_{PP}^{-1}(g_P-H_{PE}e_E),\qquad e_i=[H_{PP}^{-1}g_P]_i-\sum_{b\in\partial P}[H_{PP}^{-1}]_{ib}\sum_{j\in E}H_{bj}e_j.$$
*Proof.* The $P$-rows of $He=g$. $\square$

### 3.4 The recursive certificate

**Definition 3.4.** For each $i$ fix a patch $P_i\ni i$;
$$s_i=\big\|[H_{P_iP_i}^{-1}g_{P_i}]_i\big\|_{D_i},\qquad a_{ij}=\Big\|D_i^{1/2}\sum_{b\in\partial P_i}[H_{P_iP_i}^{-1}]_{ib}H_{bj}D_j^{-1/2}\Big\|\ (j\notin P_i),\ a_{ij}=0\ (j\in P_i).$$

**Theorem 3.5 [new, elementary].** $u_i=\|e_i\|_{D_i}$ satisfies $u\le s+Au$. If $\rho(A)<1$, $c^\star=(I-A)^{-1}s\ge0$ is finite and $c_i^\star\ge\|e_i\|_{D_i}$ for all $i$. The iteration $c^{(0)}=0$, $c^{(m+1)}=s+Ac^{(m)}$ is monotone nondecreasing and converges to $c^\star$.

*Proof.* Norms and triangle inequality in Lemma 3.3 give $u\le s+Au$. $A\ge0$, $\rho(A)<1$ give $(I-A)^{-1}=\sum A^m\ge0$; multiply $(I-A)u\le s$ by it. The iteration is the Neumann series. $\square$

$c_i$ is one scalar per node updated from exterior neighbours like a message. $\rho(A)<1$ holds when patch decay (Lemma 1.6) dominates boundary coupling: on a chain, $2\bar\psi^K<1$ suffices.

### 3.5 The chain certificate

On a block-tridiagonal chain with $P_i=[i-K,i+K]$,
$$a_{i,i\pm(K+1)}=\Big\|D_i^{1/2}[H_{P_iP_i}^{-1}]_{i,i\pm K}H_{i\pm K,i\pm(K+1)}D_{i\pm(K+1)}^{-1/2}\Big\|.$$

**Proposition 3.6 [new].** With $\bar\psi$ the maximum certified decay factor over the patch's links (seeded by Lemma 1.4 at the patch boundary) and $m$ the normalized coupling of the boundary link, $a_{i,i\pm(K+1)}\le\lambda_{\max}(D_i^{1/2}[H_{PP}^{-1}]_{ii}D_i^{1/2})\,\bar\psi^K m$, hence $\rho(A)\le2\max_i(\cdot)$. *Proof.* Lemma 1.6 applied to $H_{PP}$, itself block-tridiagonal. $\square$

### 3.6 The chain counterexample (P1)

**Proposition 3.7 [verified for the scalar model; block version to be confirmed numerically].** On a chain of $n$ nodes with every $D$-normalized link coupling of norm $\tfrac12$, $M=\tfrac12\times$ (path adjacency), $\rho(M)=\cos\frac{\pi}{n+1}\to1$, and $C=(I-M)^{-1}$ has entries of order $n$. The classical certificate and $\sqrt{C_{ii}}$ are vacuous on long prior-dominated trajectories, though the chain is a tree, GBP is exact on it, and its inverse is a product of local transfer operators.

*Proof.* Path adjacency eigenvalues $2\cos(k\pi/(n+1))$, scaled by $\tfrac12$; the scalar coupling-$\tfrac12$ chain is the Dirichlet Laplacian, whose inverse has entries of order $n$. $\square$

Why $\tfrac12$: a two-sided scalar chain with prior link precision $p$ and measurement precision $m$ has diagonal $2p+m$ and coupling $p$, so the normalized coupling is $p/(2p+m)\to\tfrac12$ as $m/p\to0$. The 12-dimensional block version does not reduce to a scalar; compute $\rho(M)$ directly on exported systems and synthetic chains.

---

## 4. Hubs

### 4.1 Supernodes and the hub set (P2)

Fix $w\ge1$. Group consecutive poses into supernodes of $w$ poses. A landmark whose observations span at most $w$ consecutive poses lies in at most two adjacent supernodes; assign it to the earlier. $T$ = all poses and all such short-track landmarks in supernodes; $S$ = everything else (long-track landmarks; one endpoint pose of each loop closure spanning more than $w$ poses).

**Lemma 4.1 [verified].** $H_{TT}$ is block-tridiagonal in supernode order. *Proof.* A factor inside $T$ touches one supernode or two adjacent ones, by construction. $\square$

### 4.2 The hub decomposition

With $H=\begin{bmatrix}H_{TT}&H_{TS}\\H_{ST}&H_{SS}\end{bmatrix}$, define
$$\tilde e_T=H_{TT}^{-1}g_T,\qquad\hat S=H_{SS}-H_{ST}H_{TT}^{-1}H_{TS},\qquad r_S=g_S-H_{ST}\tilde e_T.$$

**Theorem 4.2 [verified].** $e_S=\hat S^{-1}r_S$ and $e_T=\tilde e_T-H_{TT}^{-1}H_{TS}e_S$.

*Proof.* $T$-rows of $He=g$: $H_{TT}e_T=g_T-H_{TS}e_S$. Substituting into the $S$-rows: $H_{ST}(\tilde e_T-H_{TT}^{-1}H_{TS}e_S)+H_{SS}e_S=g_S$, i.e. $\hat Se_S=r_S$. $\square$

### 4.3 Bounding the hub error

**Lemma 4.3 [verified].** Let $H'$ be $H$ with any factors deleted, $H'\succ0$ on its variables, and $\hat S'$ the marginal precision of $S$ in $H'$. Then $\hat S'\preceq\hat S$.

*Proof.* $H'\preceq H$ (zero-padded), so $\Sigma'\succeq\Sigma$, so $\Sigma'_{SS}\succeq\Sigma_{SS}$, so $\hat S'=(\Sigma'_{SS})^{-1}\preceq\hat S$. $\square$

Take $H'$ = hub + observing poses in a window + factors among them + **the gauge anchor (P5)**; without the anchor $\hat S'$ is singular in gauge directions. Define
$$\theta_S=\lambda_{\max}\!\big(H_{SS}^{1/2}\hat S_{\mathrm{loc}}^{-1}H_{SS}^{1/2}\big)\ge1.$$

**Lemma 4.4 [new].** With $N(S)$ the supernodes observing $S$ and $m_{Sj}=\|H_{SS}^{-1/2}H_{Sj}D_j^{-1/2}\|\le1$,
$$\|e_S\|_{H_{SS}}\le\theta_S\Big(\|g_S\|_{H_{SS}^{-1}}+\sum_{j\in N(S)}m_{Sj}\|\tilde e_j\|_{D_j}\Big).$$

*Proof.* $\|e_S\|_{H_{SS}}=\|(H_{SS}^{1/2}\hat S^{-1}H_{SS}^{1/2})(H_{SS}^{-1/2}r_S)\|\le\theta_S\|r_S\|_{H_{SS}^{-1}}$ using $\hat S^{-1}\preceq\hat S_{\mathrm{loc}}^{-1}$; then $\|r_S\|_{H_{SS}^{-1}}\le\|g_S\|_{H_{SS}^{-1}}+\sum_j\|H_{SS}^{-1/2}H_{Sj}\tilde e_j\|$ with $D_j^{-1/2}D_j^{1/2}$ inserted. $m_{Sj}\le1$ because $\begin{bmatrix}D_j&H_{jS}\\H_{Sj}&H_{SS}\end{bmatrix}$ is a principal submatrix of $H\succ0$. $\square$

### 4.4 Bounding the hub's influence on a pose

**Lemma 4.5 [new].** For $i\in T$, $\Sigma^T=H_{TT}^{-1}$, $\mu^T_{\mathrm{hi},i}\ge\lambda_{\max}(D_i^{1/2}\Sigma^T_{ii}D_i^{1/2})$ (Theorem 2.4 on the chain), $m'_{jS}=\|D_j^{-1/2}H_{jS}H_{SS}^{-1/2}\|\le1$:
$$\phi_{iS}:=\Big\|D_i^{1/2}[H_{TT}^{-1}H_{TS}]_iH_{SS}^{-1/2}\Big\|\le\mu^T_{\mathrm{hi},i}\sum_{j\in N(S)}\Big(\prod_{\text{links }i\to j}\bar\psi\Big)m'_{jS},$$
with the tail beyond $K$ links at most $2\mu^T_{\mathrm{hi},i}\bar\psi^{K+1}/(1-\bar\psi)$ when $\bar\psi<1$ on it.

*Proof.* $[H_{TT}^{-1}H_{TS}]_i=\sum_j\Sigma^T_{ij}H_{jS}$; insert $D$'s and apply Lemma 1.6 per term; at most two supernodes per distance on a chain. $\square$

### 4.5 The hub-corrected certificate

**Theorem 4.6 [new].** Let $\hat c_j\ge\|\tilde e_j\|_{D_j}$ be a chain certificate (Theorem 3.5 or Proposition 3.6 on $H_{TT}$, $g_T$). Define
$$E_S=\theta_S\Big(\|g_S\|_{H_{SS}^{-1}}+\sum_{j\in N(S)}m_{Sj}\hat c_j\Big),\qquad E_i=\hat c_i+\bar\phi_{iS}E_S,$$
with $\bar\phi_{iS}$ the bound of Lemma 4.5. Then $\|e_i\|_{D_i}\le E_i$ for all $i\in T$ and $\|e_S\|_{H_{SS}}\le E_S$.

*Proof.* Theorem 4.2 and the triangle inequality: $\|e_i\|_{D_i}\le\|\tilde e_i\|_{D_i}+\phi_{iS}\|e_S\|_{H_{SS}}\le\hat c_i+\bar\phi_{iS}E_S$ by Lemma 4.4 with $\tilde e_j$ replaced by $\hat c_j$. $\square$

If $S=\varnothing$, $E_i=\hat c_i$. Several hubs: apply Lemma 4.3 jointly with an anchored subgraph containing all hubs, or treat hubs one at a time (valid, loosens with their number).

### 4.6 Hub-aware statistical upper bound

**Proposition 4.7 [new].** For $i\in T$, $\lambda_{\max}(D_i^{1/2}\Sigma_{ii}D_i^{1/2})\le\mu^T_{\mathrm{hi},i}+\theta_S\bar\phi_{iS}^2$.

*Proof.* Block inversion: $\Sigma_{TT}=H_{TT}^{-1}+H_{TT}^{-1}H_{TS}\hat S^{-1}H_{ST}H_{TT}^{-1}$, so $\Sigma_{ii}=\Sigma^T_{ii}+X\hat S^{-1}X^\top$ with $X=[H_{TT}^{-1}H_{TS}]_i$. With $Y=D_i^{1/2}XH_{SS}^{-1/2}$, $D_i^{1/2}X\hat S^{-1}X^\top D_i^{1/2}=Y(H_{SS}^{1/2}\hat S^{-1}H_{SS}^{1/2})Y^\top\preceq\theta_S\|Y\|^2I$. $\square$

The lower bound needs nothing new: Lemma 2.1 with a $T$-patch subspace is valid on the full graph.

### 4.7 What is loose

$\theta_S$ if the anchored local subgraph poorly determines the hub; the sum $\sum_jm_{Sj}\hat c_j$, which forgoes cancellation among observers' errors (the likely culprit on ChArUco); $\mu^T_{\mathrm{hi},i}$ from Theorem 2.4. If the sum is loose, replace it by the energy form $\|r_S\|_{H_{SS}^{-1}}$ computed with the chain error vector; this is a test, not a theorem here.

---

## 5. Stopping, validity over time, sleep/wake, queries

### 5.1 The stopping theorem

**Theorem 5.1 [new as a statement].** Let $E_i$ be any certificate with $\|x_i-\bar x_i\|_{D_i}\le E_i$, and $\underline S_i(\delta)\le r_i^\star(\delta)\le\overline S_i(\delta)$ from Proposition 2.7. Then (1) completing inference on the current model cannot move $x_i$ by more than $E_i$; (2) $\Pr[\|x_i-x_i^\star\|_{D_i}\le E_i+\overline S_i(\delta)]\ge1-\delta$; (3) if $E_i\le\eta\underline S_i(\delta)$, further change to $x_i$ from completing inference is at most an $\eta$-fraction of a lower bound on the exact estimator's own $(1-\delta)$ radius, so $x_i$ is within $(1+\eta)r_i^\star(\delta)$ of the truth with probability $\ge1-\delta$, and no further computation on this model can reduce that radius below $r_i^\star(\delta)$.

*Proof.* (1) definition; (2) triangle inequality and Proposition 2.7; (3) $E_i\le\eta\underline S_i\le\eta r_i^\star$. $\square$

### 5.2 Validity over iterations and relinearizations (P4)

**Theorem 5.2 [verified].** (a) Within one model, 5.1(2) holds simultaneously for every iterate $t$ and at any data-dependent stopping time, with no union bound. (b) Across $N$ relinearizations (synchronous epochs, or asynchronous snapshots at which certificates are evaluated), using $\delta/N$ per model gives simultaneous validity with probability $\ge1-\delta$.

*Proof.* (a) The event $A=\{\|\bar x_i-x_i^\star\|_{D_i}\le\overline S_i(\delta)\}$ has $\Pr[A]\ge1-\delta$ and concerns $\bar x$ only; on $A$ the bound holds for every $t$ pointwise. (b) Boole. $\square$

$N$ must be bounded in advance or $\delta$ split by a convergent series. The supermartingale route to unbounded $N$ was dropped: relinearization does not make $\bar x^{(t)}$ a martingale.

### 5.3 Sleep and wake

**Lemma 5.3 [new, elementary].** For sleeping node $i$ with patch $P$ and $\ell_{ip}=\|D_i^{1/2}[H_{PP}^{-1}]_{ip}D_p^{1/2}\|$ computed once, if the patch residual changes by $\Delta g_P$ then $s_i(g_P+\Delta g_P)\le s_i(g_P)+\sum_{p\in P}\ell_{ip}\|\Delta g_p\|_{D_p^{-1}}$, and $\Delta g_p=\sum_jH_{pj}\Delta x_j$ depends only on the moves of $p$'s neighbours. *Proof.* Linearity and the triangle inequality. $\square$

**Theorem 5.4 (certified sleep) [new].** A sleeping node maintaining $s_i(t_0)+\sum_p\ell_{ip}\|\Delta g_p\|+\sum_ja_{ij}c_j(t)$ ($+\bar\phi_{iS}E_S(t)$ with hubs) holds a valid certificate at every $t$ without recomputing its state, and must wake only when this exceeds $\eta\underline S_i$. *Proof.* Each term bounds the corresponding term of Theorem 3.5/4.6 at current data. $\square$

**Theorem 5.5 (bounded wake wave) [new].** A new factor on $V_f\subset T$ changes $g$ only on $V_f$; the change $\Delta s$ in local solves is nonzero only on nodes whose patch meets $V_f$. With $r=$ max row sum of $A$, $r<1$, at recursion distance $m$ from $\mathrm{supp}(\Delta s)$,
$$\Delta c^\star_i\le\frac{r^m}{1-r}\|\Delta s\|_\infty,$$
so a node with slack $\sigma_i=\eta\underline S_i-E_i$ stays asleep whenever $r^m\|\Delta s\|_\infty/(1-r)<\sigma_i$. The wake set lies within $O(\log(\|\Delta s\|_\infty/\sigma)/\log(1/r))$ patches of $V_f$, independent of trajectory length.

*Proof.* $\Delta c^\star=\sum_mA^m\Delta s$; $[A^m\Delta s]_i=0$ below the recursion distance; $\|A^m\|_\infty\le r^m$. $\square$

If $V_f$ contains a hub, $E_S$ changes and every observer's certificate changes by $\bar\phi_{iS}\Delta E_S$: one broadcast scalar, cost $O(\deg S)$. A loop closure adds an endpoint to $S$ and requires recomputing $\hat S_{\mathrm{loc}}$, $\theta_S$ once.

### 5.4 Continuous-time query [conditional]

Assume $x(\tau)=\Lambda(\tau)x_k+\Psi(\tau)x_{k+1}+\text{const}$ in local coordinates for $\tau\in[t_k,t_{k+1}]$ (exactly sparse GP interpolation). Confirm from the code.

**Theorem 5.6 [conditional].** For SPD $W$,
$$\|x^{(t)}(\tau)-\bar x(\tau)\|_W\le\|W^{1/2}\Lambda(\tau)D_k^{-1/2}\|E_k+\|W^{1/2}\Psi(\tau)D_{k+1}^{-1/2}\|E_{k+1}.$$
$\mathrm{Cov}(\bar x(\tau)-x^\star(\tau))=[\Lambda\ \Psi]\Sigma_{(k,k+1)}[\Lambda\ \Psi]^\top+Q_\tau$, with $\Sigma_{(k,k+1)}$ the joint endpoint marginal (bounded by Lemma 2.1 and Theorem 2.4/Proposition 4.7 on the pair as one block) and $Q_\tau$ the conditional process covariance of the interpolant; bounds push through by monotonicity of $X\mapsto TXT^\top$.

*Proof.* Affine maps commute with the error decomposition; law of total covariance for the interpolant. $\square$

### 5.5 The global check

Stack all whitened Jacobians including the anchor into $\tilde A$ so $H=\tilde A^\top\tilde A$, $b=\tilde A^\top\tilde y$.

**Proposition 5.7 [classical].** (a) $\|e\|_H^2=\|\tilde Ax-\tilde y\|^2-\|\tilde A\bar x-\tilde y\|^2\le\|\tilde Ax-\tilde y\|^2$. (b) $\|e\|_H^2=g^\top H^{-1}g\ge(g^\top g)^2/(g^\top Hg)$, and the $\ell$-step Gauss rule from Lanczos on $g$ gives larger valid lower bounds.

*Proof.* (a) $\tilde A\bar x-\tilde y\perp\mathrm{range}\,\tilde A$; Pythagoras. (b) Lemma 2.1 with $V=g/\|g\|$ and with the Krylov subspace. $\square$

(b) is a rigorous cheap lower bound on total remaining error energy. (a) is loose (the optimal cost is of order the number of measurements) and is the $\chi^2$ model-misfit monitor; not for stopping.

---

## 6. Dropped or deferred

- Supermartingale anytime validity over unbounded relinearizations: replaced by Theorem 5.2(b).
- Per-pose localization of the Prager–Synge bound: unproved and doubtful; replaced by Proposition 5.7 (global).
- Demko–Moss–Smith decay with global constants: replaced by Lemma 1.6 (local).
- Gauss–Radau upper bounds on error: need a certified $\lambda_{\min}(H)$, not locally available; only Gauss lower bounds kept.
- Newton–Kantorovich nonlinear certificate: deferred; nothing here depends on it.

## 7. Facts to read off G-solver

1. Relinearization schedule (P4). 2. Gauge fixing, including distributed mode (P5). 3. The exact interpolation operator (affine form of §5.4). 4. Whether stored backward messages equal the chain Schur complements $G_k$, so Lemma 1.6 can reuse them.
