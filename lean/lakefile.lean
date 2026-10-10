import Lake
open Lake DSL

package splitjepa where
  leanOptions := #[
    ⟨`autoImplicit, false⟩
  ]

@[default_target]
lean_lib SplitJEPA where

require "leanprover-community" / "mathlib" @ git "v4.28.0"
