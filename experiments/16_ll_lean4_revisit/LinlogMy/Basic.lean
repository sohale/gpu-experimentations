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

-- example →→→ theorem

theorem pos_mem_nonneg
 (x : ℝ) (hx : 0 < x) : x ∈ { y : ℝ | 0 ≤ y } := by
  -- unfold set membership
  -- bad: show 0 ≤ x
  change 0 ≤ x
  -- strict positivity implies nonnegativity
  exact le_of_lt hx

-- succeeds silently

#print pos_mem_nonneg
