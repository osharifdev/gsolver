# The corrected variant: true Jacobians (Stage 1)

WP6b found two factors whose Jacobian is not the derivative of their own error along the solver's charts. The drivers' `--true-jacobians` flag (global `gsolver::true_jacobians`, `gsolver/include/gsolver/core/jacobian_mode.h`) replaces both with the derivative.

**The default is off, and the default path is byte-identical:** the shipped expressions are kept verbatim in the `else` branches (`results/stage1/equivalence.txt`, 12 of 12).

## Stereo (`se3_posevel_stereo_point.cpp`)

The rotation block of the pose Jacobian is `2.0 * P_ext * skew(p_cam)` in G-solver. The derivative is `P_ext * skew(p_cam)`: central differences give a ratio of exactly 2.0 in all 11,487 factors checked, and 1.0 after the correction.

## GP prior (`derivatives.h`, `compute_se3vel_gp_jacobian`)

The pose part of the GP error is e_p = Log(Exp(dt v)⁻¹ T_i⁻¹ T_{i+1}). The exponential and logarithm are the decoupled maps of `geometry3d.h`: Exp(t, r) = (Rodrigues(r), t), with the tangent ordered [t, r]. The first knot's velocity v = [ν; ω] enters additively.

Write R_v = Exp_SO3(dt ω), M = T_i⁻¹ T_{i+1}, p = R_vᵀ(t_M − dt ν) (the translation part of e_p) and φ = Log(R_vᵀ R_M) (its rotation part). Then:

    ∂p/∂ν = −dt R_vᵀ          ∂p/∂ω = dt [p]× J_r(dt ω)
    ∂φ/∂ν = 0                 ∂φ/∂ω = −dt J_l⁻¹(φ) J_r(dt ω)

Here J_r is the SO(3) right Jacobian and J_l⁻¹ the inverse left Jacobian.
- **J_r** follows `expMapSO3`'s own branch: that map is first order below |w|² = 1e-8, so its derivative there is the identity.
- **The shipped block** (`jacob_dDinvP1invP2_de1e2_gp`) carries a coupling of order dt·[ν]× in ∂p/∂ω. That coupling belongs to the coupled SE(3) exponential, not to the map the error uses.

With the flag, `DerivativeUtils::jacob_gp_pose_error_dv_true` replaces that 6 × 6 block. Every other block of the GP Jacobian already equals the derivative.

## The true Jacobian in every export

**`exportModel` writes `f_<type>_Jtrue.npy`** for every factor, whichever variant runs: the true Jacobian at the same point as `f_<type>_J.npy`.
- In frozen charts that point is the variables' linearization points; in belief charts, the beliefs.
- Stereo: the rotation columns without the 2.
- GP: the corrected velocity block.
- Other types: the solver's Jacobian, which central differences confirm is exact for the ChArUco pose–marker and prior factors.

**`PoseModelBuilder(true_jacobian=True)`** builds the Fisher information I_F = JᵀΛJ from these arrays. That is the Stage 1 yardstick on both variants; on the corrected variant it is the solver's own Jacobian.

## Validation

`--fd-check-at R` runs `ProbeServer::fdCheck` on the live graph at the evaluation after round R. It compares every factor's Jacobian at its linearization point with central differences (h = 1e-6) of the factor's own error along the solver's charts, per variable and block of three tangent columns. `scripts/stage1_fdcheck.py` summarizes the result. The requirement is 1e-5 relative on every block of the corrected variant.
