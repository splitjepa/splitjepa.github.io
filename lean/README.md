# Lean formalization of SplitJEPA block identifiability


## Building

The project pins Lean and Mathlib to `v4.28.0`.

```bash
lake update
lake build
lake env lean SplitJEPA/Sanity.lean
```

The last command prints the axiom dependencies of every paper-facing theorem.

## Files

| File | Content |
| --- | --- |
| `SplitJEPA/Basic.lean` | The algebraic sufficient-variation condition `VariationSpans`. |
| `SplitJEPA/Ambiguity.lean` | Triangular leakage, its inverse, and Theorem 1's observational equivalence. |
| `SplitJEPA/Blocks.lean` | The two short block-elimination lemmas used by Theorem 2. |
| `SplitJEPA/Identifiability.lean` | The paper-facing block-identifiability theorem. |
| `SplitJEPA/Robustness.lean` | The finite variant-isolation form underlying Theorem 3. |
| `SplitJEPA/Sanity.lean` | Axiom audit for the public results. |
| `SplitJEPA.lean` | Public import surface. |

## Correspondence with the paper

All declarations are in the namespace `SplitJEPA`.

| Paper object or result | Lean declaration | Formalized content |
| --- | --- | --- |
| Latent split `z = (z_H, z_L)` | product type `H × L` | Separate invariant and variant coordinates. |
| Theorem 1 transformation `z̃_H = z_H + ψ(z_L)`, `z̃_L = z_L` | `leak ψ` | Triangular cross-block mixing. |
| Inverse transformed coordinates | `unleak ψ` | `z_H = z̃_H - ψ(z̃_L)`. |
| Exact invertibility | `unleak_leak` | Applying the inverse after the leak returns the original latent state. |
| Theorem 1: observational equivalence | `nonlinear_latent_ambiguity` | `(f ∘ unleak ψ) (leak ψ z) = f z`. |
| Nontrivial cross-block dependence | `leak_changes_first_block` | A nonconstant `ψ` makes the transformed first block vary with `z_L`. |
| Assumption 1: sufficient variation | `VariationSpans Δ` | The observed differences `Δz_L` span the entire variant space. |
| Invariant consistency | hypothesis `hinv` | `A₁₂ Δz_L = 0` for every matched-pair difference. |
| Elimination of `A₁₂` | `upperRight_eq_zero` | Invariant consistency on a spanning family forces `A₁₂ = 0`. |
| Orthogonality step | `lowerLeft_eq_zero` | The upper-left and cross-block equations force `A₂₁ = 0`. |
| Theorem 2 | `splitjepa_identifiability` | Both cross-blocks vanish and both diagonal blocks are orthogonal. |
| Invariant-only downstream computation | `invariantOnly` | The prediction factors through the invariant coordinate. |
| Theorem 3: variant isolation | `variant_isolation` | Any finite change in `z_L` leaves the prediction unchanged when `z_H` is fixed. |

## The main result

After predictive identification, the paper writes the recovered representation
as

```text
        [ A₁₁  A₁₂ ]
Q   =   [          ],       Q Qᵀ = I.
        [ A₂₁  A₂₂ ]
```

The Lean theorem receives exactly the three block equations of `Q Qᵀ = I`,
the invariant-consistency equation on matched-pair differences, and sufficient
variation:

```lean
theorem splitjepa_identifiability
    (A11 : Matrix (Fin dH) (Fin dH) ℝ)
    (A12 : Matrix (Fin dH) (Fin dL) ℝ)
    (A21 : Matrix (Fin dL) (Fin dH) ℝ)
    (A22 : Matrix (Fin dL) (Fin dL) ℝ)
    (Δ : Ω → Fin dL → ℝ)
    (hspan : VariationSpans Δ)
    (hinv : ∀ ω, A12.mulVec (Δ ω) = 0)
    (hupper : A11 * A11.transpose + A12 * A12.transpose = 1)
    (hcross : A11 * A21.transpose + A12 * A22.transpose = 0)
    (hlower : A21 * A21.transpose + A22 * A22.transpose = 1) :
    A12 = 0 ∧ A21 = 0 ∧
      A11 * A11.transpose = 1 ∧ A11.transpose * A11 = 1 ∧
      A22 * A22.transpose = 1 ∧ A22.transpose * A22 = 1
```

This is the block-diagonal conclusion of Theorem 2:

```text
        [ A₁₁   0  ]
Q   =   [          ],
        [  0   A₂₂ ]
```

with independent orthogonal changes of coordinates inside the invariant and
variant subspaces.

## Proof structure

The Lean proof follows Steps 2 and 3 of Appendix B.2.

1. **Invariant consistency removes the upper-right block.**  
   The equations `A₁₂ Δz_L = 0` say that the linear map represented by
   `A₁₂` vanishes on every observed variant difference. Since these
   differences span the variant space, the map is zero: `A₁₂ = 0`.

2. **Metric preservation removes the lower-left block.**  
   Substituting `A₁₂ = 0` into the upper-left block equation gives
   `A₁₁ A₁₁ᵀ = I`. Hence `A₁₁` is invertible. The cross-block equation
   reduces to `A₁₁ A₂₁ᵀ = 0`, so `A₂₁ = 0`.

3. **The diagonal blocks are orthogonal.**  
   The remaining upper-left and lower-right equations give orthogonality of
   `A₁₁` and `A₂₂` in both multiplication orders.

No decoder or observation reconstruction is used in these steps.
