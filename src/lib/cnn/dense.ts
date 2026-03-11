import { Matrix2D, DenseConfig } from '../../types/cnn';
import { randomMatrix2D } from './matrix';
import { applyActivationToArray } from './activations';

export function denseLayer(
  input: number[],
  config: DenseConfig
): number[] {
  const { units, activation } = config;
  const inputSize = input.length;

  const weights = config.weights || initializeWeights(inputSize, units);
  const bias = config.bias || Array(units).fill(0);

  const output = Array(units).fill(0);

  for (let i = 0; i < units; i++) {
    let sum = bias[i];
    for (let j = 0; j < inputSize; j++) {
      sum += input[j] * weights[j][i];
    }
    output[i] = sum;
  }

  return applyActivationToArray(output, activation);
}

function initializeWeights(inputSize: number, outputSize: number): Matrix2D {
  const limit = Math.sqrt(6 / (inputSize + outputSize));
  return randomMatrix2D(inputSize, outputSize, -limit, limit);
}

export function flatten(input: number[][][]): number[] {
  return input.flat(2);
}
