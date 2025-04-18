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

def Context := List Formula

inductive Provable : Context → Formula → Prop
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

-- #eval IO.println hello
