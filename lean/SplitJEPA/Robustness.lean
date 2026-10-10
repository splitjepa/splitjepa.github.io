import SplitJEPA.Identifiability

/-! Finite variant isolation, the global form underlying Theorem 3. -/

namespace SplitJEPA

variable {H L R Y C : Type*}

def invariantOnly (encodeInvariant : H → R) (predict : C → R → Y)
    (context : C) : H × L → Y :=
  fun z => predict context (encodeInvariant z.1)

/-- An invariant-only predictor ignores every finite variant change. -/
theorem variant_isolation
    (encodeInvariant : H → R) (predict : C → R → Y)
    (context : C) (h : H) (l₁ l₂ : L) :
    invariantOnly encodeInvariant predict context (h, l₁) =
      invariantOnly encodeInvariant predict context (h, l₂) := by
  rfl

end SplitJEPA
