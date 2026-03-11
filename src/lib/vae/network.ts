import { VAEArchitecture, LatentSpace } from '../../types/vae';

export function encode(input: number[], encoder: any): { mean: number[]; logVar: number[] } {
  let activation = input;

  for (const layer of encoder) {
    activation = dense(activation, layer.weights, layer.biases, layer.activation);
  }

  const dim = activation.length / 2;
  const mean = activation.slice(0, dim);
  const logVar = activation.slice(dim);

  return { mean, logVar };
}

export function reparameterize(mean: number[], logVar: number[]): number[] {
  const std = logVar.map(x => Math.exp(0.5 * x));
  const epsilon = std.map(() => randomNormal());
  return mean.map((m, i) => m + std[i] * epsilon[i]);
}

export function decode(latent: number[], decoder: any): number[] {
  let activation = latent;

  for (const layer of decoder) {
    activation = dense(activation, layer.weights, layer.biases, layer.activation);
  }

  return activation;
}

export function vaeForward(
  input: number[],
  architecture: VAEArchitecture
): { reconstruction: number[]; latent: LatentSpace } {
  const { mean, logVar } = encode(input, architecture.encoder);
  const sample = reparameterize(mean, logVar);
  const reconstruction = decode(sample, architecture.decoder);

  return {
    reconstruction,
    latent: {
      mean,
      logVariance: logVar,
      sample,
      dimension: mean.length
    }
  };
}

function dense(
  input: number[],
  weights: number[][],
  biases: number[],
  activation: string
): number[] {
  const z = weights.map((row, i) =>
    row.reduce((sum, w, j) => sum + w * input[j], biases[i])
  );

  return applyActivation(z, activation);
}

function applyActivation(vector: number[], activation: string): number[] {
  switch (activation) {
    case 'relu':
      return vector.map(x => Math.max(0, x));
    case 'sigmoid':
      return vector.map(x => 1 / (1 + Math.exp(-x)));
    case 'tanh':
      return vector.map(x => Math.tanh(x));
    default:
      return vector;
  }
}

function randomNormal(): number {
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}
