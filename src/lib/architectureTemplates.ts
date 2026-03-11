import { Layer } from '../types/cnn';

export interface ArchitectureTemplate {
  id: string;
  name: string;
  description: string;
  layers: Omit<Layer, 'id'>[];
}

export const architectureTemplates: ArchitectureTemplate[] = [
  {
    id: 'simple-cnn',
    name: 'Simple CNN',
    description: 'Basic 2-layer CNN for beginners',
    layers: [
      {
        type: 'conv2d',
        name: 'Conv2D-1',
        config: {
          filters: 8,
          kernelSize: 3,
          stride: 1,
          padding: 'valid',
          activation: 'relu',
        },
      },
      {
        type: 'maxpool',
        name: 'MaxPool-1',
        config: {
          poolSize: 2,
          type: 'max',
          stride: 2,
        },
      },
      {
        type: 'conv2d',
        name: 'Conv2D-2',
        config: {
          filters: 16,
          kernelSize: 3,
          stride: 1,
          padding: 'valid',
          activation: 'relu',
        },
      },
      {
        type: 'maxpool',
        name: 'MaxPool-2',
        config: {
          poolSize: 2,
          type: 'max',
          stride: 2,
        },
      },
      {
        type: 'dense',
        name: 'Dense-1',
        config: {
          units: 10,
          activation: 'relu',
        },
      },
    ],
  },
  {
    id: 'lenet-style',
    name: 'LeNet-5 Style',
    description: 'Classic architecture inspired by LeNet-5',
    layers: [
      {
        type: 'conv2d',
        name: 'Conv2D-1',
        config: {
          filters: 6,
          kernelSize: 5,
          stride: 1,
          padding: 'valid',
          activation: 'relu',
        },
      },
      {
        type: 'avgpool',
        name: 'AvgPool-1',
        config: {
          poolSize: 2,
          type: 'average',
          stride: 2,
        },
      },
      {
        type: 'conv2d',
        name: 'Conv2D-2',
        config: {
          filters: 16,
          kernelSize: 5,
          stride: 1,
          padding: 'valid',
          activation: 'relu',
        },
      },
      {
        type: 'avgpool',
        name: 'AvgPool-2',
        config: {
          poolSize: 2,
          type: 'average',
          stride: 2,
        },
      },
      {
        type: 'dense',
        name: 'Dense-1',
        config: {
          units: 120,
          activation: 'relu',
        },
      },
      {
        type: 'dense',
        name: 'Dense-2',
        config: {
          units: 84,
          activation: 'relu',
        },
      },
      {
        type: 'dense',
        name: 'Dense-3',
        config: {
          units: 10,
          activation: 'relu',
        },
      },
    ],
  },
  {
    id: 'deep-cnn',
    name: 'Deep CNN',
    description: 'Deeper network with multiple conv layers',
    layers: [
      {
        type: 'conv2d',
        name: 'Conv2D-1',
        config: {
          filters: 32,
          kernelSize: 3,
          stride: 1,
          padding: 'valid',
          activation: 'relu',
        },
      },
      {
        type: 'conv2d',
        name: 'Conv2D-2',
        config: {
          filters: 32,
          kernelSize: 3,
          stride: 1,
          padding: 'valid',
          activation: 'relu',
        },
      },
      {
        type: 'maxpool',
        name: 'MaxPool-1',
        config: {
          poolSize: 2,
          type: 'max',
          stride: 2,
        },
      },
      {
        type: 'conv2d',
        name: 'Conv2D-3',
        config: {
          filters: 64,
          kernelSize: 3,
          stride: 1,
          padding: 'valid',
          activation: 'relu',
        },
      },
      {
        type: 'conv2d',
        name: 'Conv2D-4',
        config: {
          filters: 64,
          kernelSize: 3,
          stride: 1,
          padding: 'valid',
          activation: 'relu',
        },
      },
      {
        type: 'maxpool',
        name: 'MaxPool-2',
        config: {
          poolSize: 2,
          type: 'max',
          stride: 2,
        },
      },
      {
        type: 'dense',
        name: 'Dense-1',
        config: {
          units: 128,
          activation: 'relu',
        },
      },
      {
        type: 'dense',
        name: 'Dense-2',
        config: {
          units: 10,
          activation: 'relu',
        },
      },
    ],
  },
  {
    id: 'edge-detector',
    name: 'Edge Detector',
    description: 'Specialized for edge detection',
    layers: [
      {
        type: 'conv2d',
        name: 'Conv2D-1',
        config: {
          filters: 16,
          kernelSize: 3,
          stride: 1,
          padding: 'valid',
          activation: 'relu',
        },
      },
      {
        type: 'conv2d',
        name: 'Conv2D-2',
        config: {
          filters: 32,
          kernelSize: 3,
          stride: 1,
          padding: 'valid',
          activation: 'relu',
        },
      },
      {
        type: 'maxpool',
        name: 'MaxPool-1',
        config: {
          poolSize: 2,
          type: 'max',
          stride: 2,
        },
      },
    ],
  },
];
