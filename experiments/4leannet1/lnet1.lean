import Mathlib.Data.List.Basic

-- Placeholder for a real number type, using Float for simplicity
def Real := Float

-- Library of activation functions
def relu (x : Real) : Real :=
  max 0.0 x

def leakyRelu (x : Real) (α : Real := 0.01) : Real :=
  if x > 0.0 then x else α * x

def tanh (x : Real) : Real :=
  (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x))

def sigmoid (x : Real) : Real :=
  1.0 / (1.0 + Math.exp(-x))


-- Main NN definitions

-- Define a structure for a neuron (weights and bias)
structure Neuron :=
  (weights : List Real)
  (bias : Real)

-- Define a type for a layer, which is a list of neurons
def Layer := List Neuron

-- A function to create a neuron with a given number of inputs (weights) initialized randomly
def createNeuron (numInputs : Nat) : Neuron :=
  let weights := List.replicate numInputs 0.1 -- Placeholder for random initialization
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
def activationFunction (x : Real) : Real :=
  sigmoid x

-- Function to compute the output of a neuron given its inputs
def computeNeuronOutput (neuron : Neuron) (inputs : List Real) : Real :=
  let weightedSum := List.foldl (λ acc (weightInput : Real × Real) => acc + (weightInput.1 * weightInput.2)) 0 (List.zip neuron.weights inputs)
  activationFunction (weightedSum + neuron.bias)

-- Example usage: Define the network architecture
#eval createNetwork [4, 10, 3, 1]
