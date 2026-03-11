export interface RNNCell {
  id: string;
  type: 'rnn' | 'lstm' | 'gru';
  hiddenSize: number;
  weights?: {
    input: number[][];
    hidden: number[][];
    output: number[][];
  };
}

export interface LSTMGate {
  forget: number[][];
  input: number[][];
  output: number[][];
  cell: number[][];
}

export interface RNNArchitecture {
  cells: RNNCell[];
  sequenceLength: number;
  inputSize: number;
  hiddenSize: number;
}

export interface RNNState {
  architecture: RNNArchitecture;
  hiddenStates: number[][][];
  cellStates?: number[][][];
  outputs: number[][];
}
