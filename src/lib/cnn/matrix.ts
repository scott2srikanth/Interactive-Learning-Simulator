import { Matrix2D, Matrix3D } from '../../types/cnn';

export function create2DMatrix(rows: number, cols: number, value: number = 0): Matrix2D {
  return Array(rows)
    .fill(0)
    .map(() => Array(cols).fill(value));
}

export function create3DMatrix(
  depth: number,
  rows: number,
  cols: number,
  value: number = 0
): Matrix3D {
  return Array(depth)
    .fill(0)
    .map(() => create2DMatrix(rows, cols, value));
}

export function randomMatrix2D(rows: number, cols: number, min: number = -0.5, max: number = 0.5): Matrix2D {
  return Array(rows)
    .fill(0)
    .map(() =>
      Array(cols)
        .fill(0)
        .map(() => Math.random() * (max - min) + min)
    );
}

export function padMatrix(
  matrix: Matrix2D,
  padding: number,
  value: number = 0
): Matrix2D {
  const rows = matrix.length;
  const cols = matrix[0].length;
  const padded = create2DMatrix(rows + 2 * padding, cols + 2 * padding, value);

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      padded[i + padding][j + padding] = matrix[i][j];
    }
  }

  return padded;
}

export function getOutputSize(
  inputSize: number,
  kernelSize: number,
  stride: number,
  padding: number
): number {
  return Math.floor((inputSize + 2 * padding - kernelSize) / stride) + 1;
}

export function normalize(matrix: Matrix2D): Matrix2D {
  let min = Infinity;
  let max = -Infinity;

  for (let i = 0; i < matrix.length; i++) {
    for (let j = 0; j < matrix[i].length; j++) {
      min = Math.min(min, matrix[i][j]);
      max = Math.max(max, matrix[i][j]);
    }
  }

  const range = max - min;
  if (range === 0) return matrix;

  return matrix.map(row => row.map(val => (val - min) / range));
}

export function flatten3DMatrix(matrix: Matrix3D): number[] {
  return matrix.flat(2);
}
