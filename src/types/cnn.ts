export type Matrix2D = number[][];
export type Matrix3D = number[][][];
export type Matrix4D = number[][][][];

export type ActivationType = 'relu' | 'sigmoid' | 'tanh' | 'linear';
export type PoolingType = 'max' | 'average';
export type PaddingType = 'valid' | 'same';

export interface Conv2DConfig {
  filters: number;
  kernelSize: number;
  stride: number;
  padding: PaddingType;
  activation: ActivationType;
  kernelWeights?: Matrix4D;
}

export interface PoolingConfig {
  poolSize: number;
  stride: number;
  type: PoolingType;
}

export interface DenseConfig {
  units: number;
  activation: ActivationType;
  weights?: Matrix2D;
  bias?: number[];
}

export type LayerType =
  | 'input'
  | 'conv2d'
  | 'maxpool'
  | 'avgpool'
  | 'flatten'
  | 'dense'
  | 'dropout'
  | 'softmax';

export interface BaseLayer {
  id: string;
  type: LayerType;
  name: string;
  output?: Matrix3D | number[];
}

export interface InputLayer extends BaseLayer {
  type: 'input';
  shape: [number, number, number];
}

export interface Conv2DLayer extends BaseLayer {
  type: 'conv2d';
  config: Conv2DConfig;
}

export interface PoolingLayer extends BaseLayer {
  type: 'maxpool' | 'avgpool';
  config: PoolingConfig;
}

export interface FlattenLayer extends BaseLayer {
  type: 'flatten';
}

export interface DenseLayer extends BaseLayer {
  type: 'dense';
  config: DenseConfig;
}

export interface SoftmaxLayer extends BaseLayer {
  type: 'softmax';
}

export type Layer =
  | InputLayer
  | Conv2DLayer
  | PoolingLayer
  | FlattenLayer
  | DenseLayer
  | SoftmaxLayer;

export interface CNNArchitecture {
  layers: Layer[];
  input: Matrix3D | null;
}

export interface TrainingMetrics {
  epoch: number;
  loss: number;
  accuracy: number;
}

export interface UserProgress {
  id: string;
  user_id: string;
  xp: number;
  level: number;
  completed_lessons: string[];
  earned_badges: string[];
  created_at: string;
  updated_at: string;
}

export interface Badge {
  id: string;
  name: string;
  description: string;
  icon: string;
  requirement: string;
}

export interface Challenge {
  id: string;
  title: string;
  description: string;
  difficulty: 'easy' | 'medium' | 'hard';
  xp_reward: number;
  badge_reward?: string;
  requirements: {
    maxParameters?: number;
    minAccuracy?: number;
    requiredLayers?: LayerType[];
  };
}
