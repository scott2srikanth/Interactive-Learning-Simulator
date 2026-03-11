import { AttentionHead, TransformerLayer } from '../../types/transformer';

export function scaledDotProductAttention(
  query: number[][],
  key: number[][],
  value: number[][],
  mask?: number[][]
): { output: number[][]; attention: number[][] } {
  const dK = key[0].length;
  const scores = matrixMultiply(query, transpose(key));

  const scaledScores = scores.map(row =>
    row.map(val => val / Math.sqrt(dK))
  );

  if (mask) {
    for (let i = 0; i < scaledScores.length; i++) {
      for (let j = 0; j < scaledScores[i].length; j++) {
        if (mask[i][j] === 0) {
          scaledScores[i][j] = -Infinity;
        }
      }
    }
  }

  const attention = scaledScores.map(row => softmax(row));
  const output = matrixMultiply(attention, value);

  return { output, attention };
}

export function multiHeadAttention(
  input: number[][],
  heads: AttentionHead[],
  dModel: number
): { output: number[][]; attentions: number[][][] } {
  const headOutputs: number[][][] = [];
  const attentions: number[][][] = [];

  for (const head of heads) {
    const query = matrixMultiply(input, head.queryWeights);
    const key = matrixMultiply(input, head.keyWeights);
    const value = matrixMultiply(input, head.valueWeights);

    const { output, attention } = scaledDotProductAttention(query, key, value);
    headOutputs.push(output);
    attentions.push(attention);
  }

  const concatenated = concatenateHeads(headOutputs);

  return { output: concatenated, attentions };
}

export function positionEncoding(sequenceLength: number, dModel: number): number[][] {
  const encoding: number[][] = [];

  for (let pos = 0; pos < sequenceLength; pos++) {
    const row: number[] = [];
    for (let i = 0; i < dModel; i++) {
      if (i % 2 === 0) {
        row.push(Math.sin(pos / Math.pow(10000, i / dModel)));
      } else {
        row.push(Math.cos(pos / Math.pow(10000, (i - 1) / dModel)));
      }
    }
    encoding.push(row);
  }

  return encoding;
}

function matrixMultiply(a: number[][], b: number[][]): number[][] {
  const result: number[][] = [];
  for (let i = 0; i < a.length; i++) {
    result[i] = [];
    for (let j = 0; j < b[0].length; j++) {
      let sum = 0;
      for (let k = 0; k < b.length; k++) {
        sum += a[i][k] * b[k][j];
      }
      result[i][j] = sum;
    }
  }
  return result;
}

function transpose(matrix: number[][]): number[][] {
  return matrix[0].map((_, i) => matrix.map(row => row[i]));
}

function softmax(vector: number[]): number[] {
  const expValues = vector.map(x => Math.exp(x));
  const sum = expValues.reduce((a, b) => a + b, 0);
  return expValues.map(x => x / sum);
}

function concatenateHeads(heads: number[][][]): number[][] {
  const seqLength = heads[0].length;
  const result: number[][] = [];

  for (let i = 0; i < seqLength; i++) {
    const row: number[] = [];
    for (const head of heads) {
      row.push(...head[i]);
    }
    result.push(row);
  }

  return result;
}
