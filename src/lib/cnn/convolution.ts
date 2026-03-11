import { Matrix2D, Matrix3D, Matrix4D, Conv2DConfig, PaddingType } from '../../types/cnn';
import { create2DMatrix, create3DMatrix, padMatrix, getOutputSize, randomMatrix2D } from './matrix';
import { applyActivation } from './activations';

export function convolve2D(
  input: Matrix2D,
  kernel: Matrix2D,
  stride: number = 1
): Matrix2D {
  const inputHeight = input.length;
  const inputWidth = input[0].length;
  const kernelSize = kernel.length;

  const outputHeight = Math.floor((inputHeight - kernelSize) / stride) + 1;
  const outputWidth = Math.floor((inputWidth - kernelSize) / stride) + 1;

  const output = create2DMatrix(outputHeight, outputWidth);

  for (let i = 0; i < outputHeight; i++) {
    for (let j = 0; j < outputWidth; j++) {
      let sum = 0;
      const startRow = i * stride;
      const startCol = j * stride;

      for (let ki = 0; ki < kernelSize; ki++) {
        for (let kj = 0; kj < kernelSize; kj++) {
          sum += input[startRow + ki][startCol + kj] * kernel[ki][kj];
        }
      }

      output[i][j] = sum;
    }
  }

  return output;
}

export function conv2DLayer(
  input: Matrix3D,
  config: Conv2DConfig
): Matrix3D {
  const { filters, kernelSize, stride, padding, activation } = config;

  const inputDepth = input.length;
  const inputHeight = input[0].length;
  const inputWidth = input[0][0].length;

  let paddingSize = 0;
  if (padding === 'same') {
    paddingSize = Math.floor(kernelSize / 2);
  }

  const paddedInput = input.map(channel => padMatrix(channel, paddingSize, 0));
  const paddedHeight = paddedInput[0].length;
  const paddedWidth = paddedInput[0][0].length;

  const outputHeight = getOutputSize(paddedHeight, kernelSize, stride, 0);
  const outputWidth = getOutputSize(paddedWidth, kernelSize, stride, 0);

  const kernelWeights = config.kernelWeights || initializeKernels(
    filters,
    inputDepth,
    kernelSize
  );

  const output = create3DMatrix(filters, outputHeight, outputWidth);

  for (let f = 0; f < filters; f++) {
    for (let i = 0; i < outputHeight; i++) {
      for (let j = 0; j < outputWidth; j++) {
        let sum = 0;
        const startRow = i * stride;
        const startCol = j * stride;

        for (let c = 0; c < inputDepth; c++) {
          for (let ki = 0; ki < kernelSize; ki++) {
            for (let kj = 0; kj < kernelSize; kj++) {
              sum += paddedInput[c][startRow + ki][startCol + kj] *
                     kernelWeights[f][c][ki][kj];
            }
          }
        }

        output[f][i][j] = sum;
      }
    }
  }

  return applyActivation(output, activation) as Matrix3D;
}

function initializeKernels(
  numFilters: number,
  inputDepth: number,
  kernelSize: number
): Matrix4D {
  const kernels: Matrix4D = [];
  const limit = Math.sqrt(6 / (kernelSize * kernelSize * inputDepth));

  for (let f = 0; f < numFilters; f++) {
    const filterKernels: Matrix3D = [];
    for (let c = 0; c < inputDepth; c++) {
      filterKernels.push(randomMatrix2D(kernelSize, kernelSize, -limit, limit));
    }
    kernels.push(filterKernels);
  }

  return kernels;
}

export function createEdgeDetectionKernel(): Matrix2D {
  return [
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1],
  ];
}

export function createVerticalEdgeKernel(): Matrix2D {
  return [
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1],
  ];
}

export function createHorizontalEdgeKernel(): Matrix2D {
  return [
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1],
  ];
}

export function createSharpenKernel(): Matrix2D {
  return [
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0],
  ];
}

export function createBlurKernel(): Matrix2D {
  return [
    [1 / 9, 1 / 9, 1 / 9],
    [1 / 9, 1 / 9, 1 / 9],
    [1 / 9, 1 / 9, 1 / 9],
  ];
}
