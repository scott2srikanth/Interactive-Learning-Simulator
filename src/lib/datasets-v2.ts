import { Matrix3D } from '../types/cnn';

export interface SampleImage {
  id: string;
  name: string;
  data: Matrix3D;
  label: string;
  isRGB: boolean;
}

function makeGrid(size: number): number[][] {
  return Array(size).fill(0).map(() => Array(size).fill(0));
}

export function createImage(type: string, size: number = 8): Matrix3D {
  const m = () => makeGrid(size);

  // Grayscale images (1 channel)
  if (type === 'vert') {
    const g = m();
    for (let i = 0; i < size; i++) { g[i][3] = 1; g[i][4] = 1; }
    return [g];
  }
  if (type === 'horiz') {
    const g = m();
    for (let j = 0; j < size; j++) { g[3][j] = 1; g[4][j] = 1; }
    return [g];
  }
  if (type === 'diag') {
    const g = m();
    for (let i = 0; i < size; i++) g[i][i] = 1;
    return [g];
  }
  if (type === 'cross') {
    const g = m();
    for (let i = 0; i < size; i++) { g[i][size >> 1] = 1; g[size >> 1][i] = 1; }
    return [g];
  }
  if (type === 'circle') {
    const g = m(), c = size / 2 - 0.5, r = size / 3;
    for (let i = 0; i < size; i++)
      for (let j = 0; j < size; j++)
        if (Math.abs(Math.sqrt((i - c) ** 2 + (j - c) ** 2) - r) < 1) g[i][j] = 1;
    return [g];
  }
  if (type === 'box') {
    const g = m();
    for (let i = 1; i < size - 1; i++) { g[1][i] = 1; g[size - 2][i] = 1; g[i][1] = 1; g[i][size - 2] = 1; }
    return [g];
  }

  // RGB color images (3 channels)
  if (type === 'rgb_redsq') {
    const r = m(), g = m(), b = m();
    for (let i = 2; i < 6; i++) for (let j = 2; j < 6; j++) { r[i][j] = 1; g[i][j] = 0.1; b[i][j] = 0.1; }
    return [r, g, b];
  }
  if (type === 'rgb_bluecircle') {
    const r = m(), g = m(), b = m(), cx = size / 2 - 0.5, rad = size / 3;
    for (let i = 0; i < size; i++)
      for (let j = 0; j < size; j++) {
        const d = Math.sqrt((i - cx) ** 2 + (j - cx) ** 2);
        if (d < rad) { r[i][j] = 0.1; g[i][j] = 0.3; b[i][j] = 1; }
      }
    return [r, g, b];
  }
  if (type === 'rgb_greentriangle') {
    const r = m(), g = m(), b = m();
    for (let i = 2; i < 7; i++) {
      const w = (i - 2) * 1.2, l = Math.floor(4 - w / 2), ri = Math.floor(4 + w / 2);
      for (let j = l; j <= ri; j++) if (j >= 0 && j < size) { r[i][j] = 0.1; g[i][j] = 0.9; b[i][j] = 0.2; }
    }
    return [r, g, b];
  }
  if (type === 'rgb_sunset') {
    const r = m(), g = m(), b = m();
    for (let i = 0; i < size; i++)
      for (let j = 0; j < size; j++) {
        const t = i / (size - 1);
        r[i][j] = 1 - t * 0.3; g[i][j] = 0.5 - t * 0.4; b[i][j] = 0.2 + t * 0.6;
      }
    return [r, g, b];
  }
  if (type === 'rgb_flag') {
    const r = m(), g = m(), b = m();
    for (let i = 0; i < size; i++)
      for (let j = 0; j < size; j++) {
        if (i < 3) { r[i][j] = 1; g[i][j] = 0.5; b[i][j] = 0; }
        else if (i < 5) { r[i][j] = 1; g[i][j] = 1; b[i][j] = 1; }
        else { r[i][j] = 0.1; g[i][j] = 0.6; b[i][j] = 0.2; }
      }
    return [r, g, b];
  }
  if (type === 'rgb_gradient') {
    const r = m(), g = m(), b = m();
    for (let i = 0; i < size; i++)
      for (let j = 0; j < size; j++) {
        r[i][j] = i / (size - 1);
        g[i][j] = j / (size - 1);
        b[i][j] = 1 - (i + j) / (2 * size - 2);
      }
    return [r, g, b];
  }

  // Fallback
  return [m()];
}

export const IMAGE_LIST: { id: string; name: string; rgb: boolean }[] = [
  { id: 'vert', name: 'Vertical Line', rgb: false },
  { id: 'horiz', name: 'Horizontal Line', rgb: false },
  { id: 'diag', name: 'Diagonal', rgb: false },
  { id: 'cross', name: 'Cross (+)', rgb: false },
  { id: 'circle', name: 'Circle', rgb: false },
  { id: 'box', name: 'Box', rgb: false },
  { id: 'rgb_redsq', name: '🟥 Red Square (RGB)', rgb: true },
  { id: 'rgb_bluecircle', name: '🔵 Blue Circle (RGB)', rgb: true },
  { id: 'rgb_greentriangle', name: '🟢 Green Triangle (RGB)', rgb: true },
  { id: 'rgb_sunset', name: '🌅 Sunset Gradient (RGB)', rgb: true },
  { id: 'rgb_flag', name: '🇮🇳 Tricolor Flag (RGB)', rgb: true },
  { id: 'rgb_gradient', name: '🎨 RGB Gradient (RGB)', rgb: true },
];

export const ARCHITECTURE_PRESETS: Record<string, { name: string; description: string; color: string; layers: any[] }> = {
  simple: {
    name: 'Simple CNN',
    description: 'Conv → Pool → Dense',
    color: '#3b82f6',
    layers: [
      { type: 'conv2d', name: 'CONV2D', cfg: { filters: 8, kernelSize: 3, stride: 1, padding: 'same', activation: 'relu' } },
      { type: 'maxpool', name: 'MAXPOOL', cfg: { ps: 2, st: 2 } },
      { type: 'flatten', name: 'FLATTEN', cfg: {} },
      { type: 'dense', name: 'DENSE', cfg: { units: 10, activation: 'relu' } },
      { type: 'softmax', name: 'SOFTMAX', cfg: {} },
    ],
  },
  lenet: {
    name: 'LeNet-5',
    description: 'Conv5→AvgPool→Conv5→AvgPool→FC',
    color: '#8b5cf6',
    layers: [
      { type: 'conv2d', name: 'CONV2D', cfg: { filters: 6, kernelSize: 5, stride: 1, padding: 'same', activation: 'relu' } },
      { type: 'avgpool', name: 'AVGPOOL', cfg: { ps: 2, st: 2 } },
      { type: 'conv2d', name: 'CONV2D', cfg: { filters: 16, kernelSize: 5, stride: 1, padding: 'same', activation: 'relu' } },
      { type: 'avgpool', name: 'AVGPOOL', cfg: { ps: 2, st: 2 } },
      { type: 'flatten', name: 'FLATTEN', cfg: {} },
      { type: 'dense', name: 'DENSE', cfg: { units: 120, activation: 'relu' } },
      { type: 'dense', name: 'DENSE', cfg: { units: 84, activation: 'relu' } },
      { type: 'dense', name: 'DENSE', cfg: { units: 10, activation: 'relu' } },
      { type: 'softmax', name: 'SOFTMAX', cfg: {} },
    ],
  },
  resnet: {
    name: 'ResNet-style',
    description: '3 residual blocks + GAP',
    color: '#f59e0b',
    layers: [
      { type: 'conv2d', name: 'CONV2D', cfg: { filters: 16, kernelSize: 3, stride: 1, padding: 'same', activation: 'relu' } },
      { type: 'conv2d', name: 'CONV2D', cfg: { filters: 16, kernelSize: 3, stride: 1, padding: 'same', activation: 'relu' } },
      { type: 'conv2d', name: 'CONV2D', cfg: { filters: 16, kernelSize: 3, stride: 1, padding: 'same', activation: 'relu' } },
      { type: 'maxpool', name: 'MAXPOOL', cfg: { ps: 2, st: 2 } },
      { type: 'conv2d', name: 'CONV2D', cfg: { filters: 32, kernelSize: 3, stride: 1, padding: 'same', activation: 'relu' } },
      { type: 'conv2d', name: 'CONV2D', cfg: { filters: 32, kernelSize: 3, stride: 1, padding: 'same', activation: 'relu' } },
      { type: 'maxpool', name: 'MAXPOOL', cfg: { ps: 2, st: 2 } },
      { type: 'conv2d', name: 'CONV2D', cfg: { filters: 64, kernelSize: 3, stride: 1, padding: 'same', activation: 'relu' } },
      { type: 'avgpool', name: 'AVGPOOL', cfg: { ps: 2, st: 2 } },
      { type: 'flatten', name: 'FLATTEN', cfg: {} },
      { type: 'dense', name: 'DENSE', cfg: { units: 64, activation: 'relu' } },
      { type: 'dense', name: 'DENSE', cfg: { units: 10, activation: 'relu' } },
      { type: 'softmax', name: 'SOFTMAX', cfg: {} },
    ],
  },
  deep: {
    name: 'Deep CNN',
    description: '4×Conv → 2×Pool → Dense',
    color: '#06b6d4',
    layers: [
      { type: 'conv2d', name: 'CONV2D', cfg: { filters: 32, kernelSize: 3, stride: 1, padding: 'same', activation: 'relu' } },
      { type: 'conv2d', name: 'CONV2D', cfg: { filters: 32, kernelSize: 3, stride: 1, padding: 'same', activation: 'relu' } },
      { type: 'maxpool', name: 'MAXPOOL', cfg: { ps: 2, st: 2 } },
      { type: 'conv2d', name: 'CONV2D', cfg: { filters: 64, kernelSize: 3, stride: 1, padding: 'same', activation: 'relu' } },
      { type: 'conv2d', name: 'CONV2D', cfg: { filters: 64, kernelSize: 3, stride: 1, padding: 'same', activation: 'relu' } },
      { type: 'maxpool', name: 'MAXPOOL', cfg: { ps: 2, st: 2 } },
      { type: 'flatten', name: 'FLATTEN', cfg: {} },
      { type: 'dense', name: 'DENSE', cfg: { units: 128, activation: 'relu' } },
      { type: 'dense', name: 'DENSE', cfg: { units: 10, activation: 'relu' } },
      { type: 'softmax', name: 'SOFTMAX', cfg: {} },
    ],
  },
  edge: {
    name: 'Edge Detector',
    description: 'Single Conv layer',
    color: '#22c55e',
    layers: [
      { type: 'conv2d', name: 'CONV2D', cfg: { filters: 4, kernelSize: 3, stride: 1, padding: 'same', activation: 'relu' } },
    ],
  },
};
