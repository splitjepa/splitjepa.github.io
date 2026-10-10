import SplitJEPA.Blocks

/-! Reconstruction-free block identifiability (Theorem 2). -/

noncomputable section

namespace SplitJEPA

variable {Ω : Type*} {dH dL : ℕ}

/--
The block equations are the part of global orthogonality used by SplitJEPA;
the global recovery result itself is the imported LeJEPA premise.
-/
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
      A22 * A22.transpose = 1 ∧ A22.transpose * A22 = 1 := by
  have hA12 := upperRight_eq_zero A12 Δ hspan hinv
  have h11 : A11 * A11.transpose = 1 := by simpa [hA12] using hupper
  have hcross' : A11 * A21.transpose = 0 := by simpa [hA12] using hcross
  have hA21 := lowerLeft_eq_zero A11 A21 h11 hcross'
  have h22 : A22 * A22.transpose = 1 := by simpa [hA21] using hlower
  exact ⟨hA12, hA21, h11, mul_eq_one_comm.mp h11,
    h22, mul_eq_one_comm.mp h22⟩

end SplitJEPA
