import { Layer, Matrix3D, CNNArchitecture } from '../../types/cnn';
import { conv2DLayer } from './convolution';
import { poolingLayer } from './pooling';
import { flatten, denseLayer } from './dense';
import { softmax } from './activations';

export function executeForwardPass(
  architecture: CNNArchitecture,
  input: Matrix3D
): Map<string, Matrix3D | number[]> {
  const layerOutputs = new Map<string, Matrix3D | number[]>();

  let currentOutput: Matrix3D | number[] = input;
  layerOutputs.set('input', input);

  for (const layer of architecture.layers) {
    if (layer.type === 'input') {
      continue;
    }

    try {
      currentOutput = processLayer(layer, currentOutput);
      layerOutputs.set(layer.id, currentOutput);
    } catch (error) {
      console.error(`Error processing layer ${layer.id}:`, error);
      throw error;
    }
  }

  return layerOutputs;
}

function processLayer(
  layer: Layer,
  input: Matrix3D | number[]
): Matrix3D | number[] {
  switch (layer.type) {
    case 'conv2d':
      if (!Array.isArray(input[0][0])) {
        throw new Error('Conv2D layer expects 3D input');
      }
      return conv2DLayer(input as Matrix3D, layer.config);

    case 'maxpool':
      if (!Array.isArray(input[0][0])) {
        throw new Error('Pooling layer expects 3D input');
      }
      return poolingLayer(input as Matrix3D, layer.config);

    case 'avgpool':
      if (!Array.isArray(input[0][0])) {
        throw new Error('Pooling layer expects 3D input');
      }
      return poolingLayer(input as Matrix3D, layer.config);

    case 'flatten':
      if (!Array.isArray(input[0][0])) {
        throw new Error('Input already flattened');
      }
      return flatten(input as Matrix3D);

    case 'dense':
      if (Array.isArray(input[0]) && Array.isArray(input[0][0])) {
        throw new Error('Dense layer expects 1D input (use Flatten first)');
      }
      return denseLayer(input as number[], layer.config);

    case 'softmax':
      if (Array.isArray(input[0]) && Array.isArray(input[0][0])) {
        throw new Error('Softmax expects 1D input');
      }
      return softmax(input as number[]);

    default:
      return input;
  }
}

export function calculateTotalParameters(architecture: CNNArchitecture): number {
  let total = 0;

  for (const layer of architecture.layers) {
    if (layer.type === 'conv2d') {
      const { filters, kernelSize } = layer.config;
      const prevDepth = getPreviousDepth(architecture, layer.id);
      total += filters * prevDepth * kernelSize * kernelSize + filters;
    } else if (layer.type === 'dense') {
      const { units } = layer.config;
      const inputSize = getPreviousSize(architecture, layer.id);
      total += inputSize * units + units;
    }
  }

  return total;
}

function getPreviousDepth(architecture: CNNArchitecture, layerId: string): number {
  const layerIndex = architecture.layers.findIndex(l => l.id === layerId);
  if (layerIndex === 0) return 1;

  for (let i = layerIndex - 1; i >= 0; i--) {
    const prevLayer = architecture.layers[i];
    if (prevLayer.type === 'conv2d') {
      return prevLayer.config.filters;
    } else if (prevLayer.type === 'input') {
      return (prevLayer as any).shape[0];
    }
  }

  return 1;
}

function getPreviousSize(architecture: CNNArchitecture, layerId: string): number {
  const layerIndex = architecture.layers.findIndex(l => l.id === layerId);
  if (layerIndex === 0) return 0;

  const prevLayer = architecture.layers[layerIndex - 1];
  if (prevLayer.type === 'flatten') {
    return 512;
  }

  return 128;
}
