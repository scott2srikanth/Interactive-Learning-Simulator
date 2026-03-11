import { ANNLayer, ANNArchitecture } from '../../types/ann';

export function initializeWeights(inputSize: number, outputSize: number): number[][] {
  const weights: number[][] = [];
  const scale = Math.sqrt(2.0 / inputSize);

  for (let i = 0; i < outputSize; i++) {
    weights[i] = [];
    for (let j = 0; j < inputSize; j++) {
      weights[i][j] = (Math.random() * 2 - 1) * scale;
    }
  }

  return weights;
}

export function initializeBiases(size: number): number[] {
  return new Array(size).fill(0);
}

export function forwardPropagate(
  input: number[],
  architecture: ANNArchitecture
): number[][] {
  const activations: number[][] = [input];

  for (let i = 1; i < architecture.layers.length; i++) {
    const layer = architecture.layers[i];
    const prevActivation = activations[i - 1];
    const z = matrixVectorMultiply(layer.weights!, prevActivation);
    const a = applyActivation(vectorAdd(z, layer.biases!), layer.activation!);
    activations.push(a);
  }

  return activations;
}

function matrixVectorMultiply(matrix: number[][], vector: number[]): number[] {
  return matrix.map(row =>
    row.reduce((sum, val, idx) => sum + val * vector[idx], 0)
  );
}

function vectorAdd(a: number[], b: number[]): number[] {
  return a.map((val, idx) => val + b[idx]);
}

function applyActivation(vector: number[], activation: string): number[] {
  switch (activation) {
    case 'sigmoid':
      return vector.map(x => 1 / (1 + Math.exp(-x)));
    case 'relu':
      return vector.map(x => Math.max(0, x));
    case 'tanh':
      return vector.map(x => Math.tanh(x));
    case 'softmax':
      const expValues = vector.map(x => Math.exp(x));
      const sum = expValues.reduce((a, b) => a + b, 0);
      return expValues.map(x => x / sum);
    default:
      return vector;
  }
}
