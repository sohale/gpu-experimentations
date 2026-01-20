import Mathlib
-- import Paperproof
-- import Linlog1proj

#eval [1, 2] ++ [3, 4]

#check Nat
#check Int
#check Real


def hello := "world"

-- We alredy defined a main in LinlogMy/simply_ll1.lean
-- def main : IO Unit :=
--   IO.println s!"linear logoic installed"


---------

-- Now, a simple proof:

open Set

example (x : ℝ) (hx : 0 < x) : x ∈ { y : ℝ | 0 ≤ y } := by
  -- unfold set membership
  show 0 ≤ x
  -- strict positivity implies nonnegativity
  exact le_of_lt hx
