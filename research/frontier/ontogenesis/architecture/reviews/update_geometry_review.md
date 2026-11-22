# Independent review: two-dimensional update geometry

**Date:** November 21, 2026, America/New_York. **Verdict:** no defect found within the stated mathematical scope. No model experiment was run. Reviewed source identities:

| File | SHA-256 |
|---|---|
| analysis/update_feasibility_geometry.py | 2b789e2a1f809ea9a0432a3d837722cd4e05425349be0f89f5921f891e046c1e |
| checks/test_update_feasibility_geometry.py | 21a19038dca17dbe1e4f3ab5153616fe2b91986a8da54f25ca967fffc185eeda |

The source's exact rational calculation correctly separates a target margin requirement, a protected margin requirement, the choice of edit norm and tie behavior. The example is fixed at initial point (2,1), target constraint y−x ≥ 3, protected constraint x−(2/5)y ≥ μ and μ ≥ 0. Its conclusions are not universal claims about neural capacity, optimizer superiority or preservation of every learned fact.

## Independently checked derivation

For the target-only problem, set a=x−2 and b=y−1. Feasibility requires b−a ≥ 4. The triangle inequality gives |a|+|b| ≥ b−a ≥ 4. Equality requires b−a=4, a≤0 and b≥0, which is exactly the segment (x,x+3) with −2≤x≤2.

On this segment the protected margin equals (3/5)x−6/5. It is strictly negative for x<2 and zero at the endpoint (2,5). Thus every minimum-L1 target edit loses the protected **strict** winner, but the tied endpoint need not change an argmax answer under a tie-breaking rule that retains C. The source and tests correctly make that distinction.

Adding the protected constraint gives

$$
b-a\ge4,\qquad a-\tfrac25b\ge\mu-\tfrac85.
$$

Combining the lower bound b≥a+4 with the second inequality yields a≥5μ/3 and then b≥4+5μ/3. For μ≥0 both displacements are nonnegative. Therefore the objective is at least 4+10μ/3, attained at (x,y)=(2+5μ/3,5+5μ/3). This proves the claimed minimum for every real μ≥0, independently of the finite rational fixtures.

At μ=0 the optimum is the tie endpoint. If the requirement is merely a strictly positive protected margin without a fixed positive lower bound, the infimum is 4 and is not attained: all strictly feasible margins cost more than 4, while the sequence μ→0+ approaches it.

The target gradient direction (−1,+1), started from (2,1), first reaches y−x=3 at (0,3), with protected margin −6/5. This is correctly labeled an ideal geometric hitting point rather than a recorded finite-step trajectory of a named optimizer.

## Code and validation

The implementation uses Fraction for exact input and output, rejects negative or inexact margin arguments, and scopes its formula to μ≥0. The independent epigraph LP check formulates absolute displacements with auxiliary variables and reproduces the optimum for five finite fixtures. All seven checks passed when run by this reviewer. The source's CLI uses exclusive output creation.

The tests are numerical and finite checks of an elementary proof; they do not provide machine-checked formal verification of all real-valued cases. The source says so. The example depends on these coordinates and the L1 metric, and supplies neither a general capacity lower bound nor a requirement to replay every protected fact on every later update.

**Requested changes:** none. Preserve the tie caveat, the μ≥0 scope, the distinction between proof and numerical checks, and the separation from empirical optimizer claims when incorporating this result into the architecture.
