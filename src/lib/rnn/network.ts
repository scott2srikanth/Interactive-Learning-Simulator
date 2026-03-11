import { RNNCell, RNNArchitecture } from '../../types/rnn';

export function initializeRNNWeights(inputSize: number, hiddenSize: number) {
  return {
    input: randomMatrix(hiddenSize, inputSize),
    hidden: randomMatrix(hiddenSize, hiddenSize),
    output: randomMatrix(inputSize, hiddenSize)
  };
}

export function initializeLSTMWeights(inputSize: number, hiddenSize: number) {
  return {
    forget: randomMatrix(hiddenSize, inputSize + hiddenSize),
    input: randomMatrix(hiddenSize, inputSize + hiddenSize),
    output: randomMatrix(hiddenSize, inputSize + hiddenSize),
    cell: randomMatrix(hiddenSize, inputSize + hiddenSize)
  };
}

export function rnnForward(
  input: number[][],
  cell: RNNCell,
  initialHidden: number[]
): { hidden: number[][]; outputs: number[][] } {
  const hidden: number[][] = [initialHidden];
  const outputs: number[][] = [];

  for (let t = 0; t < input.length; t++) {
    const h = matrixVectorMultiply(cell.weights!.input, input[t]);
    const hPrev = matrixVectorMultiply(cell.weights!.hidden, hidden[t]);
    const newHidden = vectorAdd(h, hPrev).map(x => Math.tanh(x));
    hidden.push(newHidden);

    const output = matrixVectorMultiply(cell.weights!.output, newHidden);
    outputs.push(output);
  }

  return { hidden: hidden.slice(1), outputs };
}

export function lstmForward(
  input: number[][],
  gates: any,
  initialHidden: number[],
  initialCell: number[]
): { hidden: number[][]; cell: number[][]; outputs: number[][] } {
  const hidden: number[][] = [initialHidden];
  const cell: number[][] = [initialCell];
  const outputs: number[][] = [];

  for (let t = 0; t < input.length; t++) {
    const combined = [...input[t], ...hidden[t]];

    const forgetGate = sigmoid(matrixVectorMultiply(gates.forget, combined));
    const inputGate = sigmoid(matrixVectorMultiply(gates.input, combined));
    const outputGate = sigmoid(matrixVectorMultiply(gates.output, combined));
    const cellCandidate = tanh(matrixVectorMultiply(gates.cell, combined));

    const newCell = vectorAdd(
      vectorMultiply(forgetGate, cell[t]),
      vectorMultiply(inputGate, cellCandidate)
    );

    const newHidden = vectorMultiply(outputGate, tanh(newCell));

    hidden.push(newHidden);
    cell.push(newCell);
    outputs.push(newHidden);
  }

  return {
    hidden: hidden.slice(1),
    cell: cell.slice(1),
    outputs
  };
}

function randomMatrix(rows: number, cols: number): number[][] {
  return Array(rows).fill(0).map(() =>
    Array(cols).fill(0).map(() => Math.random() * 0.2 - 0.1)
  );
}

function matrixVectorMultiply(matrix: number[][], vector: number[]): number[] {
  return matrix.map(row =>
    row.reduce((sum, val, idx) => sum + val * vector[idx], 0)
  );
}

function vectorAdd(a: number[], b: number[]): number[] {
  return a.map((val, idx) => val + b[idx]);
}

function vectorMultiply(a: number[], b: number[]): number[] {
  return a.map((val, idx) => val * b[idx]);
}

function sigmoid(vector: number[]): number[] {
  return vector.map(x => 1 / (1 + Math.exp(-x)));
}

function tanh(vector: number[] | number): number[] | number {
  if (Array.isArray(vector)) {
    return vector.map(x => Math.tanh(x));
  }
  return Math.tanh(vector);
}
