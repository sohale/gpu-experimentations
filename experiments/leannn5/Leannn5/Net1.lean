import Mathlib.Data.List.Basic

-- Placeholder for a real number type, using Float for simplicity
-- def Real := Float

-- Library of activation functions
def relu (x : Float) : Float :=
  max 0.0 x

def leakyRelu (x : Float) (α : Float := 0.01) : Float :=
  if x > 0.0 then x else α * x

def tanh (x : Float) : Float :=
  (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x))

def sigmoid (x : Float) : Float :=
  1.0 / (1.0 + Math.exp(-x))


-- Main NN definitions

-- Define a structure for a neuron (weights and bias)
structure Neuron :=
  (weights : List Float)
  (bias : Float)

-- Define a type for a layer, which is a list of neurons
def Layer := List Neuron

-- A function to create a neuron with a given number of inputs (weights) initialized randomly
def createNeuron (numInputs : Nat) : Neuron :=
  let weights := List.replicate numInputs 0.0 -- Placeholder for random initialization
  let bias := 0.1 -- Placeholder for random initialization
  Neuron.mk weights bias

-- Function to create a layer with a specified number of neurons, each having the same number of inputs
def createLayer (numNeurons numInputs : Nat) : Layer :=
  List.replicate numNeurons (createNeuron numInputs)

-- Function to initialize the network architecture based on the provided list of layer sizes
def createNetwork (layerSizes : List Nat) : List Layer :=
  match layerSizes with
  | [] => []
  | _ :: t => List.zipWith createLayer t layerSizes

-- Placeholder for the activation function, to be defined based on the specific use case
def activationFunction (x : Float) : Float :=
  sigmoid x

-- Function to compute the output of a neuron given its inputs
def computeNeuronOutput (neuron : Neuron) (inputs : List Float) : Float :=
  let weightedSum := List.foldl (λ acc (weightInput : Float × Float) => acc + (weightInput.1 * weightInput.2)) 0 (List.zip neuron.weights inputs)
  activationFunction (weightedSum + neuron.bias)


/-
instance : Repr Neuron where
  reprPrec n _ :=
    "Neuron(" ++ toString n.weights ++ ", " ++ toString n.bias ++ ")"

instance : Repr Layer where
  reprPrec l _ :=
    "Layer(" ++ toString l.neurons ++ ")"
-/

-- (λ w => if w == 0 then "." else toString w) weight

instance : Repr Neuron where
  reprPrec neuron _ :=
    let weightsStr := String.intercalate ", " (neuron.weights.map (λ w => if w == 0 then "₀" else toString w))
    s!"Neuron(weights: [{ weightsStr}], bias: {neuron.bias})"

-- Example usage: Define the network architecture
def n := createNetwork [4, 10, 3, 1]

#eval n
