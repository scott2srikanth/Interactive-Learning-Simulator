export interface ANNLayer {
  id: string;
  type: 'input' | 'hidden' | 'output';
  neurons: number;
  activation?: 'sigmoid' | 'relu' | 'tanh' | 'softmax';
  weights?: number[][];
  biases?: number[];
}

export interface ANNArchitecture {
  layers: ANNLayer[];
  learningRate: number;
}

export interface ANNState {
  architecture: ANNArchitecture;
  activations: number[][];
  isTraining: boolean;
}
