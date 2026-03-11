import { Matrix2D, Matrix3D } from '../../types/cnn';

export function conv2DWithPadding(input: Matrix2D, kernel: Matrix2D, stride: number = 1, padding: number = 0): Matrix2D {
  let padded = input;
  if (padding > 0) {
    const pH = input.length + 2 * padding;
    const pW = input[0].length + 2 * padding;
    padded = Array(pH).fill(0).map(() => Array(pW).fill(0));
    for (let i = 0; i < input.length; i++)
      for (let j = 0; j < input[0].length; j++)
        padded[i + padding][j + padding] = input[i][j];
  }
  const k = kernel.length;
  const oH = Math.floor((padded.length - k) / stride) + 1;
  const oW = Math.floor((padded[0].length - k) / stride) + 1;
  const out: Matrix2D = [];
  for (let i = 0; i < oH; i++) {
    const row: number[] = [];
    for (let j = 0; j < oW; j++) {
      let s = 0;
      for (let a = 0; a < k; a++)
        for (let b = 0; b < k; b++)
          s += (padded[i * stride + a]?.[j * stride + b] || 0) * kernel[a][b];
      row.push(s);
    }
    out.push(row);
  }
  return out;
}

export function conv3D(channels: Matrix3D, kernel: Matrix2D, stride: number = 1, padding: number = 0): Matrix2D {
  let summed: Matrix2D | null = null;
  for (let c = 0; c < channels.length; c++) {
    const result = conv2DWithPadding(channels[c], kernel, stride, padding);
    if (!summed) {
      summed = result;
    } else {
      for (let i = 0; i < result.length; i++)
        for (let j = 0; j < result[0].length; j++)
          summed[i][j] += result[i][j];
    }
  }
  return summed!;
}

export function applyRelu(m: Matrix2D): Matrix2D {
  return m.map(r => r.map(v => Math.max(0, v)));
}

export function maxPool(input: Matrix2D, poolSize: number = 2, stride: number = 2): Matrix2D {
  const oH = Math.floor((input.length - poolSize) / stride) + 1;
  const oW = Math.floor((input[0].length - poolSize) / stride) + 1;
  const out: Matrix2D = [];
  for (let i = 0; i < oH; i++) {
    const row: number[] = [];
    for (let j = 0; j < oW; j++) {
      let mx = -Infinity;
      for (let a = 0; a < poolSize; a++)
        for (let b = 0; b < poolSize; b++) {
          const v = input[i * stride + a]?.[j * stride + b] ?? 0;
          if (v > mx) mx = v;
        }
      row.push(mx);
    }
    out.push(row);
  }
  return out;
}

export function avgPool(input: Matrix2D, poolSize: number = 2, stride: number = 2): Matrix2D {
  const oH = Math.floor((input.length - poolSize) / stride) + 1;
  const oW = Math.floor((input[0].length - poolSize) / stride) + 1;
  const out: Matrix2D = [];
  for (let i = 0; i < oH; i++) {
    const row: number[] = [];
    for (let j = 0; j < oW; j++) {
      let s = 0;
      for (let a = 0; a < poolSize; a++)
        for (let b = 0; b < poolSize; b++)
          s += input[i * stride + a]?.[j * stride + b] ?? 0;
      row.push(s / (poolSize * poolSize));
    }
    out.push(row);
  }
  return out;
}

export function flattenChannels(channels: Matrix3D | number[][]): number[] {
  const flat: number[] = [];
  channels.forEach(c => {
    if (Array.isArray(c[0])) {
      (c as number[][]).forEach(r => r.forEach(v => flat.push(v)));
    } else {
      (c as number[]).forEach(v => flat.push(v));
    }
  });
  return flat;
}

export function softmax(arr: number[]): number[] {
  const mx = Math.max(...arr);
  const exps = arr.map(v => Math.exp(v - mx));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / sum);
}

export const KERNEL_PRESETS: Record<string, { name: string; kernel: Matrix2D }> = {
  edge: { name: 'Edge Detection', kernel: [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]] },
  vert_edge: { name: 'Vertical Edge', kernel: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] },
  horiz_edge: { name: 'Horizontal Edge', kernel: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]] },
  sharpen: { name: 'Sharpen', kernel: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]] },
  blur: { name: 'Box Blur', kernel: [[1, 1, 1], [1, 1, 1], [1, 1, 1]].map(r => r.map(v => +(v / 9).toFixed(3))) },
  emboss: { name: 'Emboss', kernel: [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]] },
  identity: { name: 'Identity', kernel: [[0, 0, 0], [0, 1, 0], [0, 0, 0]] },
};

export function getKernelForLayer(layerId: string): Matrix2D {
  const seed = layerId.split('').reduce((a, c) => a + c.charCodeAt(0), 0);
  const pool = Object.values(KERNEL_PRESETS);
  return pool[seed % pool.length].kernel;
}

export interface LayerOutput {
  type: '3d' | '1d' | 'error';
  data: Matrix3D | number[] | null;
  error?: string;
}

export interface SimpleLayer {
  id: string;
  type: string;
  name: string;
  cfg: Record<string, any>;
}

export function executeForwardPass(layers: SimpleLayer[], inputImage: Matrix3D): Map<string, LayerOutput> {
  const outs = new Map<string, LayerOutput>();
  let cur: any = inputImage;

  for (const L of layers) {
    try {
      if (L.type === 'conv2d') {
        const ker = getKernelForLayer(L.id);
        const nf = Math.min(L.cfg.filters || 4, 6);
        const inChs = (Array.isArray(cur[0]) && Array.isArray(cur[0][0])) ? cur : [cur];
        const pad = L.cfg.padding === 'same' ? Math.floor(ker.length / 2) : 0;
        const chs: Matrix2D[] = [];
        for (let f = 0; f < nf; f++) {
          const result = conv3D(inChs, ker, L.cfg.stride || 1, pad);
          chs.push(L.cfg.activation === 'relu' ? applyRelu(result) : result);
        }
        cur = chs;
        outs.set(L.id, { type: '3d', data: cur });
      } else if (L.type === 'maxpool') {
        cur = cur.map((c: Matrix2D) => maxPool(c, L.cfg.ps || 2, L.cfg.st || 2));
        outs.set(L.id, { type: '3d', data: cur });
      } else if (L.type === 'avgpool') {
        cur = cur.map((c: Matrix2D) => avgPool(c, L.cfg.ps || 2, L.cfg.st || 2));
        outs.set(L.id, { type: '3d', data: cur });
      } else if (L.type === 'flatten') {
        const f = flattenChannels(cur);
        cur = [f];
        outs.set(L.id, { type: '1d', data: f });
      } else if (L.type === 'dense') {
        const inp = Array.isArray(cur[0]) && Array.isArray(cur[0][0])
          ? flattenChannels(cur)
          : cur.flat ? cur.flat() : cur[0] || [];
        const units = L.cfg.units || 10;
        const r: number[] = [];
        for (let i = 0; i < units; i++) {
          let s = 0;
          for (let j = 0; j < Math.min(inp.length, 20); j++)
            s += inp[j] * (Math.sin(i * 13.7 + j * 7.3) * 0.3);
          r.push(L.cfg.activation === 'relu' ? Math.max(0, s) : s);
        }
        cur = [r];
        outs.set(L.id, { type: '1d', data: r });
      } else if (L.type === 'softmax') {
        const inp = Array.isArray(cur[0]) && Array.isArray(cur[0][0])
          ? flattenChannels(cur)
          : cur.flat ? cur.flat() : cur[0] || [];
        const sm = softmax(inp.slice(0, 10));
        cur = [sm];
        outs.set(L.id, { type: '1d', data: sm });
      }
    } catch (e: any) {
      outs.set(L.id, { type: 'error', data: null, error: e.message });
    }
  }
  return outs;
}
