export interface VAELayer {
  id: string;
  type: 'encoder' | 'latent' | 'decoder';
  size: number;
  activation?: string;
}

export interface LatentSpace {
  mean: number[];
  logVariance: number[];
  sample: number[];
  dimension: number;
}

export interface VAEArchitecture {
  encoder: VAELayer[];
  latentDim: number;
  decoder: VAELayer[];
}

export interface VAEState {
  architecture: VAEArchitecture;
  latentSpace: LatentSpace;
  reconstruction: number[];
  encoderOutputs: number[][];
  decoderOutputs: number[][];
}
