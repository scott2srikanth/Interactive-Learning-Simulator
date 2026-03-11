import { create } from 'zustand';
import { Layer, Matrix3D, CNNArchitecture, TrainingMetrics } from '../types/cnn';
import { executeForwardPass, calculateTotalParameters } from '../lib/cnn/forwardPass';

interface CNNState {
  architecture: CNNArchitecture;
  selectedLayerId: string | null;
  layerOutputs: Map<string, Matrix3D | number[]>;
  isSimulating: boolean;
  trainingMetrics: TrainingMetrics[];

  setArchitecture: (architecture: CNNArchitecture) => void;
  addLayer: (layer: Layer) => void;
  removeLayer: (layerId: string) => void;
  updateLayer: (layerId: string, updates: Partial<Layer>) => void;
  selectLayer: (layerId: string | null) => void;
  setInput: (input: Matrix3D) => void;
  runSimulation: () => void;
  runForwardPass: () => Promise<void>;
  clearArchitecture: () => void;
  getTotalParameters: () => number;
  simulateTraining: (epochs: number) => void;
}

export const useCNNStore = create<CNNState>((set, get) => ({
  architecture: {
    layers: [],
    input: null,
  },
  selectedLayerId: null,
  layerOutputs: new Map(),
  isSimulating: false,
  trainingMetrics: [],

  setArchitecture: (architecture) => set({ architecture }),

  addLayer: (layer) =>
    set((state) => ({
      architecture: {
        ...state.architecture,
        layers: [...state.architecture.layers, layer],
      },
    })),

  removeLayer: (layerId) =>
    set((state) => ({
      architecture: {
        ...state.architecture,
        layers: state.architecture.layers.filter((l) => l.id !== layerId),
      },
      selectedLayerId: state.selectedLayerId === layerId ? null : state.selectedLayerId,
    })),

  updateLayer: (layerId, updates) =>
    set((state) => ({
      architecture: {
        ...state.architecture,
        layers: state.architecture.layers.map((l) =>
          l.id === layerId ? { ...l, ...updates } : l
        ),
      },
    })),

  selectLayer: (layerId) => set({ selectedLayerId: layerId }),

  setInput: (input) =>
    set((state) => ({
      architecture: {
        ...state.architecture,
        input,
      },
    })),

  runSimulation: () => {
    const { architecture } = get();

    if (!architecture.input || architecture.layers.length === 0) {
      return;
    }

    set({ isSimulating: true });

    try {
      const outputs = executeForwardPass(architecture, architecture.input);
      set({ layerOutputs: outputs, isSimulating: false });
    } catch (error) {
      console.error('Simulation error:', error);
      set({ isSimulating: false });
    }
  },

  runForwardPass: async () => {
    const { architecture } = get();

    if (!architecture.input || architecture.layers.length === 0) {
      return;
    }

    set({ isSimulating: true });

    try {
      await new Promise((resolve) => setTimeout(resolve, 100));
      const outputs = executeForwardPass(architecture, architecture.input);
      set({ layerOutputs: outputs, isSimulating: false });
    } catch (error) {
      console.error('Forward pass error:', error);
      set({ isSimulating: false });
    }
  },

  clearArchitecture: () =>
    set({
      architecture: { layers: [], input: null },
      selectedLayerId: null,
      layerOutputs: new Map(),
      trainingMetrics: [],
    }),

  getTotalParameters: () => {
    const { architecture } = get();
    return calculateTotalParameters(architecture);
  },

  simulateTraining: (epochs: number) => {
    const { architecture } = get();
    const complexity = architecture.layers.length;
    const parameters = get().getTotalParameters();

    const metrics: TrainingMetrics[] = [];

    for (let epoch = 0; epoch < epochs; epoch++) {
      const progress = epoch / epochs;

      const baseLoss = 2.3;
      const lossDecay = Math.exp(-3 * progress);
      const loss = baseLoss * lossDecay * (1 + Math.random() * 0.1);

      const baseAccuracy = 0.1;
      const accuracyGrowth = 0.85 * (1 - Math.exp(-3 * progress));
      const complexityBonus = Math.min(complexity * 0.02, 0.1);
      const accuracy = Math.min(
        baseAccuracy + accuracyGrowth + complexityBonus + (Math.random() * 0.05 - 0.025),
        0.98
      );

      metrics.push({
        epoch: epoch + 1,
        loss: parseFloat(loss.toFixed(4)),
        accuracy: parseFloat(accuracy.toFixed(4)),
      });
    }

    set({ trainingMetrics: metrics });
  },
}));
