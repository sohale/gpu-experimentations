-- Lean 4
-- Lean 4 version of the simply typed lambda calculus
-- with linear logic
-- https://leanprover.github.io/lean4/doc/lean4.html
-- https://leanprover.github.io/lean4/doc/tutorials/lean4_tutorial.html

set_option diagnostics true


-- multiplicative fragment (⊗, ⅋, 1, ⊥).

-- syntax of formulas

inductive Formula
| one    : Formula               -- 1
| bot    : Formula               -- ⊥
| tensor : Formula → Formula → Formula  -- A ⊗ B
| par    : Formula → Formula → Formula  -- A ⅋ B


/-
"Context":
In sequent calculi (like for Linear Logic),
a context (eg. Γ) typically refers to the multiset or list of assumptions
on the left-hand side of the turnstile (⊢), i.e.: Γ ⊢ A
-/
-- sequents:
-- sequent: sequent as a list of formulas on the left-hand side proving a single formula on the right

-- def Context := List Formula
-- inductive Provable : List Formula → Formula → Prop

-- inductive Provable : Context → Formula → Prop
inductive Provable : List Formula → Formula → Prop
| ax      : ∀ A, Provable [A] A
| oneR    : Provable [] Formula.one
| botL    : ∀ Γ A, Provable Γ A → Provable (Formula.bot :: Γ) A
| tensorR : ∀ Γ₁ Γ₂ A B,
             Provable Γ₁ A →
             Provable Γ₂ B →
             Provable (Γ₁ ++ Γ₂) (Formula.tensor A B)
| parL    : ∀ Γ A B C,
             Provable (A :: B :: Γ) C →
             Provable (Formula.par A B :: Γ) C

-- try proving:
--    ⊢ (1 ⅋ ⊥) ⊗ 1

-- unknown identifier 'hello'
-- #eval IO.println hello

#eval [1, 2] ++ [3, 4]

-- A `--run` cli will need a Main: "unknown declaration 'main'"

def main1 : IO Nat := do
  IO.println "Fine. Everything compiles and runs fine."
  return 42

-- cannot be main: List String -> IO UInt32
def main2 : IO Nat := do
  return 42

def main : IO Unit := do
  do
    IO.println "Fine. "
    IO.println "Everything compiles and runs fine."


-- runs twice!
#eval main

#eval main1
#eval main2

-- yay
#check 1
#check 1+2
