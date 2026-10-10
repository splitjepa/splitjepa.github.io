import SplitJEPA.Basic

/-! The two block-elimination steps used by Theorem 2. -/

noncomputable section

namespace SplitJEPA

variable {Ω : Type*} {dH dL : ℕ}

lemma upperRight_eq_zero
    (A12 : Matrix (Fin dH) (Fin dL) ℝ) (Δ : Ω → Fin dL → ℝ)
    (hspan : VariationSpans Δ) (hinv : ∀ ω, A12.mulVec (Δ ω) = 0) :
    A12 = 0 := by
  have hlin : A12.mulVecLin = 0 := by
    apply LinearMap.ext_on_range hspan
    intro ω
    simpa using hinv ω
  ext i j
  have h := congrArg (fun T => T (Pi.single j 1)) hlin
  have hij := congrFun h i
  simpa only [Matrix.mulVecLin_apply, Matrix.mulVec_single_one,
    LinearMap.zero_apply, Pi.zero_apply] using hij

lemma lowerLeft_eq_zero
    (A11 : Matrix (Fin dH) (Fin dH) ℝ)
    (A21 : Matrix (Fin dL) (Fin dH) ℝ)
    (h11 : A11 * A11.transpose = 1)
    (hcross : A11 * A21.transpose = 0) :
    A21 = 0 := by
  have h11' : A11.transpose * A11 = 1 := mul_eq_one_comm.mp h11
  have ht : A21.transpose = 0 := by
    calc
      A21.transpose = (1 : Matrix (Fin dH) (Fin dH) ℝ) * A21.transpose := by
        rw [Matrix.one_mul]
      _ = (A11.transpose * A11) * A21.transpose := by rw [h11']
      _ = A11.transpose * (A11 * A21.transpose) := by rw [Matrix.mul_assoc]
      _ = 0 := by rw [hcross, Matrix.mul_zero]
  simpa using congrArg Matrix.transpose ht

end SplitJEPA
