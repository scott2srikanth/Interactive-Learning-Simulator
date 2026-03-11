export interface AttentionHead {
  id: string;
  queryWeights: number[][];
  keyWeights: number[][];
  valueWeights: number[][];
  attentionScores?: number[][];
}

export interface TransformerLayer {
  id: string;
  type: 'encoder' | 'decoder';
  numHeads: number;
  dModel: number;
  dFF: number;
  attentionHeads: AttentionHead[];
}

export interface TransformerArchitecture {
  layers: TransformerLayer[];
  vocabSize: number;
  maxSequenceLength: number;
  embeddingDim: number;
}

export interface TransformerState {
  architecture: TransformerArchitecture;
  attentionMaps: number[][][];
  embeddings: number[][];
  outputs: number[][];
}
