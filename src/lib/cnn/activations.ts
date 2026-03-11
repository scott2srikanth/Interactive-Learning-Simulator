import { Matrix2D, Matrix3D, ActivationType } from '../../types/cnn';

export function relu(x: number): number {
  return Math.max(0, x);
}

export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

export function tanh(x: number): number {
  return Math.tanh(x);
}

export function linear(x: number): number {
  return x;
}

export function applyActivation(
  matrix: Matrix2D | Matrix3D,
  activation: ActivationType
): Matrix2D | Matrix3D {
  const activationFn = getActivationFunction(activation);

  if (Array.isArray(matrix[0][0])) {
    return (matrix as Matrix3D).map(channel =>
      channel.map(row => row.map(activationFn))
    );
  } else {
    return (matrix as Matrix2D).map(row => row.map(activationFn));
  }
}

export function applyActivationToArray(
  array: number[],
  activation: ActivationType
): number[] {
  const activationFn = getActivationFunction(activation);
  return array.map(activationFn);
}

function getActivationFunction(activation: ActivationType): (x: number) => number {
  switch (activation) {
    case 'relu':
      return relu;
    case 'sigmoid':
      return sigmoid;
    case 'tanh':
      return tanh;
    case 'linear':
      return linear;
    default:
      return linear;
  }
}

export function softmax(values: number[]): number[] {
  const maxVal = Math.max(...values);
  const exps = values.map(v => Math.exp(v - maxVal));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sumExps);
}
