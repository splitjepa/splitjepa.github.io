import SplitJEPA.Basic

/-! The algebraic content of the paper's nonlinear latent ambiguity. -/

namespace SplitJEPA

variable {H L X : Type*} [AddCommGroup H]

def leak (ψ : L → H) (z : H × L) : H × L :=
  (z.1 + ψ z.2, z.2)

def unleak (ψ : L → H) (z : H × L) : H × L :=
  (z.1 - ψ z.2, z.2)

@[simp] theorem unleak_leak (ψ : L → H) (z : H × L) :
    unleak ψ (leak ψ z) = z := by
  simp [leak, unleak]

/-- Theorem 1: the transformed decoder produces the same observation. -/
theorem nonlinear_latent_ambiguity (f : H × L → X) (ψ : L → H) (z : H × L) :
    (f ∘ unleak ψ) (leak ψ z) = f z := by
  simp

/-- A nonconstant leak makes the first block depend on the variant block. -/
theorem leak_changes_first_block (ψ : L → H) {l₁ l₂ : L}
    (hψ : ψ l₁ ≠ ψ l₂) (h : H) :
    (leak ψ (h, l₁)).1 ≠ (leak ψ (h, l₂)).1 := by
  intro heq
  apply hψ
  change h + ψ l₁ = h + ψ l₂ at heq
  exact add_left_cancel heq

end SplitJEPA
