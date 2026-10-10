import Mathlib

/-! Shared definition for the SplitJEPA identifiability proof. -/

noncomputable section

namespace SplitJEPA

variable {Ω : Type*} {dL : ℕ}

/-- The observed variant differences span every variant direction. -/
def VariationSpans (Δ : Ω → Fin dL → ℝ) : Prop :=
  Submodule.span ℝ (Set.range Δ) = ⊤

end SplitJEPA
