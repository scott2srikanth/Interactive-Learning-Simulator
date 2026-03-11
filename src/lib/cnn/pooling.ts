import { Matrix2D, Matrix3D, PoolingConfig, PoolingType } from '../../types/cnn';
import { create2DMatrix, create3DMatrix, getOutputSize } from './matrix';

export function maxPool2D(
  input: Matrix2D,
  poolSize: number,
  stride: number
): Matrix2D {
  const inputHeight = input.length;
  const inputWidth = input[0].length;

  const outputHeight = Math.floor((inputHeight - poolSize) / stride) + 1;
  const outputWidth = Math.floor((inputWidth - poolSize) / stride) + 1;

  const output = create2DMatrix(outputHeight, outputWidth);

  for (let i = 0; i < outputHeight; i++) {
    for (let j = 0; j < outputWidth; j++) {
      let max = -Infinity;
      const startRow = i * stride;
      const startCol = j * stride;

      for (let pi = 0; pi < poolSize; pi++) {
        for (let pj = 0; pj < poolSize; pj++) {
          max = Math.max(max, input[startRow + pi][startCol + pj]);
        }
      }

      output[i][j] = max;
    }
  }

  return output;
}

export function avgPool2D(
  input: Matrix2D,
  poolSize: number,
  stride: number
): Matrix2D {
  const inputHeight = input.length;
  const inputWidth = input[0].length;

  const outputHeight = Math.floor((inputHeight - poolSize) / stride) + 1;
  const outputWidth = Math.floor((inputWidth - poolSize) / stride) + 1;

  const output = create2DMatrix(outputHeight, outputWidth);

  for (let i = 0; i < outputHeight; i++) {
    for (let j = 0; j < outputWidth; j++) {
      let sum = 0;
      const startRow = i * stride;
      const startCol = j * stride;

      for (let pi = 0; pi < poolSize; pi++) {
        for (let pj = 0; pj < poolSize; pj++) {
          sum += input[startRow + pi][startCol + pj];
        }
      }

      output[i][j] = sum / (poolSize * poolSize);
    }
  }

  return output;
}

export function poolingLayer(
  input: Matrix3D,
  config: PoolingConfig
): Matrix3D {
  const { poolSize, stride, type } = config;
  const poolFn = type === 'max' ? maxPool2D : avgPool2D;

  return input.map(channel => poolFn(channel, poolSize, stride));
}
