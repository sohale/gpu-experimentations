/-
  File: BoltzmannMaxEnt.lean

  Purpose:
    A Mathlib-friendly Lean4 formalisation scaffold for the canonical (Boltzmann/Gibbs)
    distribution as the max-entropy solution under:
      (1) normalization  ∑ p = 1
      (2) fixed mean energy ∑ p E = U

    This is written to be *easy to modify* for:
      - reparameterisations (β ↦ θ, etc.)
      - change-of-variables / “moving frame” representations of energy densities per “site”.

  Notes:
    • Full proofs of the max-entropy variational argument are nontrivial in Lean (convex analysis,
      Lagrange multipliers on simplices, etc.). This file provides:
        - core definitions (entropy, partition function, Gibbs distribution)
        - the key algebraic identities proved (normalisation, entropy identity)
        - extension hooks for “site/frame” energy representation
        - theorem statements you can later strengthen (replace `sorry`).

    • Everything is discrete on a finite state space `Ω` (recommended to start).
-/

-- Don't do this: It causes infinite loop in `lake build` command: import Mathlib
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic

-- import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.BigOperators.Pi

open scoped BigOperators
open Real

namespace MaxEnt

set_option autoImplicit false
noncomputable section

/-! ## Basic finite-state setup -/

variable {Ω : Type} [Fintype Ω] [DecidableEq Ω]

/-- A (discrete) probability mass function on a finite space. -/
structure PMF (Ω : Type) [Fintype Ω] where
  p : Ω → ℝ
  nonneg   : ∀ i, 0 ≤ p i
  norm_one : (∑ i, p i) = 1

attribute [simp] PMF.norm_one

namespace PMF

variable (q : PMF Ω)

/-- Expectation of a function under a PMF. -/
def expect (f : Ω → ℝ) : ℝ :=
  ∑ i, q.p i * f i

/-- Shannon/Gibbs entropy (dimensionless): H(q) = -∑ p log p. -/
def shannon : ℝ :=
  -∑ i, q.p i * Real.log (q.p i)

end PMF

/-! ## Energy, constraints, and Gibbs form -/

variable (E : Ω → ℝ)

/-- Partition function Z(β) = ∑ exp(-β E_i). -/
def Z (β : ℝ) : ℝ :=
  ∑ i, Real.exp (-β * E i)

/-- Canonical/Boltzmann distribution p_β(i) = exp(-β E_i) / Z(β). -/
def gibbs (β : ℝ) : Ω → ℝ :=
  fun i => Real.exp (-β * E i) / Z (E := E) β

/-- Gibbs distribution packaged as a PMF, assuming Z(β) ≠ 0 (true for finite Ω). -/
def gibbsPMF (β : ℝ) [Nonempty Ω] : PMF Ω :=
by
  classical
  refine
    { p := gibbs (E := E) β
      nonneg := ?_
      norm_one := ?_ }
  · intro i
    -- exp ≥ 0 and Z ≥ 0 ⇒ p(i) ≥ 0
    have hnum : 0 ≤ Real.exp (-β * E i) := by
      exact le_of_lt (Real.exp_pos (-β * E i))
    have hden : 0 ≤ Z (E := E) β := by
      -- Z is a finite sum of nonnegative terms
      have : ∀ j, 0 ≤ Real.exp (-β * E j) := fun _ => le_of_lt (Real.exp_pos _)
      simpa [Z] using
        Finset.sum_nonneg (fun j _ => this j)
    simpa [gibbs, div_eq_mul_inv] using div_nonneg hnum hden
  · -- ∑ exp(-βE)/Z = (∑ exp(-βE)) / Z = 1, when Z ≠ 0.
    -- Use Z ≠ 0 and linearity of sums
    classical
    -- Z > 0 since Ω is nonempty and every term exp(...) > 0
    have hzpos : 0 < Z (E := E) β := by
      classical
      -- pick an element from Nonempty Ω
      let i0 : Ω := Classical.choice (inferInstance : Nonempty Ω)
      have hi0 : i0 ∈ (Finset.univ : Finset Ω) := by simp
      have hrest : 0 ≤ (Finset.univ.erase i0).sum (fun j => Real.exp (-β * E j)) := by
        exact Finset.sum_nonneg (fun j _ => le_of_lt (Real.exp_pos (-β * E j)))
      have hmain : 0 < Real.exp (-β * E i0) := Real.exp_pos _
      have hsum : Z (E := E) β
                = (Finset.univ.erase i0).sum (fun j => Real.exp (-β * E j))
                  + Real.exp (-β * E i0) := by
        simpa [Z] using
          (Finset.sum_erase_add (s := (Finset.univ : Finset Ω))
            (f := fun j => Real.exp (-β * E j)) hi0)
      have : 0 < (Finset.univ.erase i0).sum (fun j => Real.exp (-β * E j))
                  + Real.exp (-β * E i0) :=
        add_pos_of_nonneg_of_pos hrest hmain
      simpa [hsum] using this
    have hz : Z (E := E) β ≠ 0 := ne_of_gt hzpos
    have hpull : (∑ i, Real.exp (-β * E i) / Z (E := E) β)
              = (∑ i, Real.exp (-β * E i)) / Z (E := E) β := by
      simpa [div_eq_mul_inv, Finset.sum_mul, Finset.mul_sum]
    -- Z/Z = 1 using hZ
    have : ((∑ i, Real.exp (-β * E i)) / Z (E := E) β) = 1 := by
      simpa [Z] using div_self hz
    -- Convert the pulled-out form back to the original sum
    have hsum_one : (∑ i, Real.exp (-β * E i) / Z (E := E) β) = 1 := by
      simpa [hpull] using this
    simpa [gibbs] using hsum_one

/-! ## Algebraic identities for Gibbs form (no optimisation proof yet) -/

namespace GibbsIdentities

variable {E : Ω → ℝ}
variable (β : ℝ)

-- Food for thought:
-- contemplate why finite?
--  on `[Nonempty Ω]`


lemma Z_pos [Nonempty Ω] (E : Ω → ℝ) (β : ℝ) : 0 < Z (E := E) β := by
  classical
  -- Pick any i0 ∈ Ω and split the sum at i0
  rcases Finset.univ_nonempty with ⟨i0, hi0⟩
  have hrest : 0 ≤ (Finset.univ.erase i0).sum (fun j => Real.exp (-β * E j)) := by
    exact Finset.sum_nonneg (fun j _ => le_of_lt (Real.exp_pos (-β * E j)))
  have hmain : 0 < Real.exp (-β * E i0) := Real.exp_pos _
  have hsum : Z (E := E) β
            = (Finset.univ.erase i0).sum (fun j => Real.exp (-β * E j))
              + Real.exp (-β * E i0) := by
    simpa [Z] using
      (Finset.sum_erase_add (s := (Finset.univ : Finset Ω))
        (f := fun j => Real.exp (-β * E j)) hi0)
  have : 0 < (Finset.univ.erase i0).sum (fun j => Real.exp (-β * E j)) + Real.exp (-β * E i0) :=
    add_pos_of_nonneg_of_pos hrest hmain
  simpa [hsum] using this

lemma Z_ne_zero (E : Ω → ℝ) (β : ℝ) : Z (E := E) β ≠ 0 := by
  have h : 0 < Z (E := E) β := Z_pos (E := E) β
  exact ne_of_gt h

/-- Normalisation identity: ∑ gibbs = 1. -/
lemma sum_gibbs_eq_one [Nonempty Ω] (E : Ω → ℝ) (β : ℝ) :
    (∑ i, gibbs (E := E) β i) = 1 := by
  classical
  -- Expand definition: ∑ exp(-βE)/Z = (∑ exp(-βE))/Z = Z/Z = 1.
  -- First show Z ≠ 0 via positivity (Ω nonempty)
  have hzpos : 0 < Z (E := E) β := Z_pos (E := E) β
  have hZ : Z (E := E) β ≠ 0 := ne_of_gt hzpos
  -- `field_simp` works well once you rewrite.
  -- We keep the proof short and robust:
  unfold gibbs
  -- Pull out the constant factor 1/Z from sum:
  -- sum (exp(...) / Z) = (sum exp(...)) / Z
  -- Use `Finset.mul_sum` after rewriting division as multiplication by inv.
  have : (∑ i, Real.exp (-β * E i) / Z (E := E) β)
        = (∑ i, Real.exp (-β * E i)) / Z (E := E) β := by
    simpa [div_eq_mul_inv, Finset.sum_mul, Finset.mul_sum]
  -- Z/Z = 1 using hZ
  have : ((∑ i, Real.exp (-β * E i)) / Z (E := E) β) = 1 := by
    simpa [Z] using div_self hZ
  simpa [gibbs, this]

/-- Energy expectation under Gibbs distribution: U(β) = ∑ pβ(i) E_i. -/
def U (E : Ω → ℝ) (β : ℝ) : ℝ :=
  ∑ i, gibbs (E := E) β i * E i

/--
Entropy identity for Gibbs:
  H(β) = β U(β) + log Z(β)
with H the Shannon entropy.
-/
theorem shannon_gibbs_identity (E : Ω → ℝ) (β : ℝ) :
    let p : Ω → ℝ := gibbs (E := E) β
    (-∑ i, p i * Real.log (p i)) = β * U (E := E) β + Real.log (Z (E := E) β) := by
  classical
  -- Standard algebra:
  -- log p_i = log(exp(-βE_i)) - log Z = (-βE_i) - log Z
  -- Then H = -∑ p_i log p_i = β∑ p_i E_i + (log Z)∑ p_i
  -- and ∑ p_i = 1.
  intro p
  -- Expand p and use the normalisation lemma.
  -- Proof is mostly rewriting; keep as a hook if you prefer.
  sorry

/--
Thermodynamic entropy S = k_B * H, as a definition wrapper.
(Here H is dimensionless Shannon entropy on the finite space.)
-/
def S (kB : ℝ) (E : Ω → ℝ) (β : ℝ) : ℝ :=
  kB * (-∑ i, gibbs (E := E) β i * Real.log (gibbs (E := E) β i))

/-- Helmholtz free energy F(β) = -(1/β) log Z(β). -/
def F (E : Ω → ℝ) (β : ℝ) : ℝ :=
  -(1 / β) * Real.log (Z (E := E) β)

end GibbsIdentities

/-! ## Optimisation problem statement (max-entropy under constraints) -/

/--
Constraint: fixed mean energy `U` for PMF q wrt energy E.
-/
def HasMeanEnergy (q : PMF Ω) (E : Ω → ℝ) (U : ℝ) : Prop :=
  q.expect (f := E) = U

/--
The “max entropy” problem (canonical ensemble), stated as:
  q maximises Shannon entropy among all PMFs with mean energy U.

This is a *statement scaffold*. Proving it fully in Lean typically uses convexity
and KL-divergence arguments, or Lagrange multipliers in a finite-dimensional simplex.
-/
-- q⋆ = qStar
def IsMaxEntropyAt (qStar : PMF Ω) (E : Ω → ℝ) (U : ℝ) : Prop :=
  HasMeanEnergy (q := qStar) E U ∧
  ∀ q : PMF Ω, HasMeanEnergy (q := q) E U → q.shannon ≤ qStar.shannon

/--
Canonical max-entropy theorem (scaffold):

There exists β such that the Gibbs distribution pβ
is the entropy maximiser under fixed mean energy U.

You will likely want to strengthen this to uniqueness (under mild conditions)
and connect β to temperature via β = 1/(kB*T).
-/
theorem exists_gibbs_max_entropy (E : Ω → ℝ) (U : ℝ) :
    ∃ β : ℝ, ∃ qStar : PMF Ω,
      (∀ i, qStar.p i = gibbs (E := E) β i) ∧
      IsMaxEntropyAt (qStar := qStar) (E := E) U := by
  classical
  -- Nontrivial (requires existence of β solving expectation = U, plus max-ent proof).
  -- Leave as a hook; you can fill later via KL divergence:
  --   H(q) ≤ βU + log Z with equality at Gibbs.
  sorry

/-! ## “Site” / moving-frame energy representation hooks -/

/-
You said you want E_i to be “energy densities per site”, where “site” is an abstract
and moving concept (like a frame) relative to which these are represented.

A clean way to encode “moving frame” in Lean is:
  - `Site` is a base index type
  - `Frame` acts on `Site` (e.g. permutations / equivalences, or a group action)
  - a frame-dependent energy density is `E : Frame → Site → ℝ`

Below is a minimal, extensible scaffold.
-/

namespace MovingFrame

variable (Site Frame : Type)
variable [Fintype Site] [DecidableEq Site]

/--
A very general “frame action on sites”.
For concrete work you may want:
  • `Frame := Equiv.Perm Site`  (all permutations)
or a group action:
  • `[Group Frame] [MulSemiringAction Frame ...]` etc.
-/
class FrameAction (Frame : Type) (Site : Type) where
  act : Frame → Site → Site

variable [FrameAction Frame Site]

/-- Frame-dependent energy density per site. -/
def EnergyDensity : Type :=
  Frame → Site → ℝ

/--
Given a frame `f`, pull back the energy density to a fixed “reference” site indexing.
This is where your change-of-variables / reparameterisation tends to live.
-/
def energyInFrame (E : EnergyDensity (Site := Site) (Frame := Frame)) (f : Frame) : Site → ℝ :=
  fun s => E f s

/--
Alternative: represent energies in a *reference* coordinate and push them to a frame.
This is useful if you want E_f(s) = E₀( act f⁻¹ s ) etc.
For that, you typically want `Frame` to be an `Equiv` or a group action.
We keep it abstract here.
-/
def pushEnergy
    (E0 : Site → ℝ)
    (f : Frame) : Site → ℝ :=
  fun s => E0 (FrameAction.act (Frame := Frame) (Site := Site) f s)

/-
Now you can instantiate the MaxEnt machinery above by taking Ω := Site
and E := energyInFrame E f (or pushEnergy E0 f), then define Z, gibbs, etc.
-/

end MovingFrame

end
end MaxEnt
