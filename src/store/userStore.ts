import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { Badge, Challenge } from '../types/cnn';

interface UserState {
  xp: number;
  level: number;
  completedLessons: string[];
  earnedBadges: string[];
  completedChallenges: string[];
  isAuthenticated: boolean;

  addXP: (amount: number) => void;
  completeLesson: (lessonId: string) => void;
  earnBadge: (badgeId: string) => void;
  completeChallenge: (challengeId: string) => void;
  setAuthenticated: (value: boolean) => void;
  resetProgress: () => void;
}

export const useUserStore = create<UserState>()(
  persist(
    (set, get) => ({
      xp: 0,
      level: 1,
      completedLessons: [],
      earnedBadges: [],
      completedChallenges: [],
      isAuthenticated: false,

      addXP: (amount) => {
        const newXP = get().xp + amount;
        set({ xp: newXP, level: Math.floor(newXP / 100) + 1 });
      },

      completeLesson: (lessonId) => {
        const { completedLessons } = get();
        if (!completedLessons.includes(lessonId)) {
          set({ completedLessons: [...completedLessons, lessonId] });
          get().addXP(50);
        }
      },

      earnBadge: (badgeId) => {
        const { earnedBadges } = get();
        if (!earnedBadges.includes(badgeId)) {
          set({ earnedBadges: [...earnedBadges, badgeId] });
          get().addXP(100);
        }
      },

      completeChallenge: (challengeId) => {
        const { completedChallenges } = get();
        if (!completedChallenges.includes(challengeId)) {
          set({ completedChallenges: [...completedChallenges, challengeId] });
        }
      },

      setAuthenticated: (value) => set({ isAuthenticated: value }),

      resetProgress: () =>
        set({ xp: 0, level: 1, completedLessons: [], earnedBadges: [], completedChallenges: [] }),
    }),
    { name: 'nn-learn-storage' }
  )
);

// ============================
// BADGES — across all topics
// ============================
export const badges: Badge[] = [
  // CNN badges
  { id: 'cnn-first-network', name: 'CNN Builder', description: 'Build your first CNN architecture', icon: '🔍', requirement: 'Create a CNN with at least 3 layers' },
  { id: 'cnn-edge-detector', name: 'Edge Detector', description: 'Use edge detection in CNN Lab', icon: '🖼️', requirement: 'Apply edge detection kernel' },
  { id: 'cnn-rgb-master', name: 'RGB Master', description: 'Process an RGB color image through CNN', icon: '🌈', requirement: 'Run forward pass on RGB image' },
  // ANN badges
  { id: 'ann-first-network', name: 'Neural Pioneer', description: 'Build your first ANN architecture', icon: '🧠', requirement: 'Create an ANN and run forward pass' },
  { id: 'ann-xor-solver', name: 'XOR Solver', description: 'Train an ANN to solve the XOR problem', icon: '⚡', requirement: 'Achieve 90%+ accuracy on XOR' },
  { id: 'ann-boundary-master', name: 'Boundary Master', description: 'Train on all 6 ANN datasets', icon: '🎯', requirement: 'Complete all ANN datasets' },
  // RNN badges
  { id: 'rnn-sequence-reader', name: 'Sequence Reader', description: 'Process your first sequence through RNN', icon: '🔄', requirement: 'Run RNN forward pass' },
  { id: 'rnn-lstm-expert', name: 'LSTM Expert', description: 'Explore all LSTM gates step by step', icon: '🧬', requirement: 'Complete LSTM step-by-step walkthrough' },
  // VAE badges
  { id: 'vae-encoder', name: 'Latent Explorer', description: 'Explore the VAE latent space', icon: '🗺️', requirement: 'Use the latent space explorer' },
  { id: 'vae-interpolator', name: 'Smooth Interpolator', description: 'Interpolate between two patterns in VAE', icon: '✨', requirement: 'Use interpolation feature' },
  // Transformer badges
  { id: 'tf-attention', name: 'Attention Seeker', description: 'Visualize self-attention in Transformers', icon: '⚡', requirement: 'View attention heatmaps' },
  { id: 'tf-multihead', name: 'Multi-Head Master', description: 'Explore multi-head attention patterns', icon: '👁️', requirement: 'Compare multiple attention heads' },
  // Cross-topic badges
  { id: 'scholar', name: 'Scholar', description: 'Complete all lessons in any topic', icon: '📚', requirement: 'Complete 6 lessons in one topic' },
  { id: 'polymath', name: 'Polymath', description: 'Complete lessons in all 5 topics', icon: '🎓', requirement: 'At least 1 lesson per topic' },
  { id: 'deep-learner', name: 'Deep Learner', description: 'Complete all 30 lessons across all topics', icon: '🏆', requirement: 'Complete every lesson' },
];

// ============================
// CHALLENGES — across all topics
// ============================
export const challenges: Challenge[] = [
  // CNN challenges
  { id: 'cnn-ch-1', title: 'Build a LeNet-5', description: 'Recreate the classic LeNet-5 architecture in the CNN Lab', difficulty: 'easy', xp_reward: 150, badge_reward: 'cnn-first-network', requirements: { requiredLayers: ['conv2d', 'avgpool', 'dense'] } },
  { id: 'cnn-ch-2', title: 'RGB Convolution', description: 'Process an RGB color image through a CNN pipeline', difficulty: 'medium', xp_reward: 200, badge_reward: 'cnn-rgb-master', requirements: {} },
  { id: 'cnn-ch-3', title: 'Deep ResNet', description: 'Build a ResNet-style architecture with 10+ layers', difficulty: 'hard', xp_reward: 300, badge_reward: 'cnn-edge-detector', requirements: { requiredLayers: ['conv2d', 'maxpool', 'dense'] } },
  // ANN challenges
  { id: 'ann-ch-1', title: 'Solve XOR', description: 'Train an ANN to 90%+ accuracy on the XOR problem', difficulty: 'easy', xp_reward: 150, badge_reward: 'ann-xor-solver', requirements: { minAccuracy: 0.9 } },
  { id: 'ann-ch-2', title: 'Spiral Classifier', description: 'Train a deep ANN to classify the spiral dataset', difficulty: 'hard', xp_reward: 300, badge_reward: 'ann-boundary-master', requirements: { minAccuracy: 0.85 } },
  // RNN challenges
  { id: 'rnn-ch-1', title: 'Sequence Predictor', description: 'Process a sine wave through RNN and observe hidden states', difficulty: 'easy', xp_reward: 150, badge_reward: 'rnn-sequence-reader', requirements: {} },
  { id: 'rnn-ch-2', title: 'LSTM Gate Explorer', description: 'Walk through all 6 LSTM gate steps for a sequence', difficulty: 'medium', xp_reward: 200, badge_reward: 'rnn-lstm-expert', requirements: {} },
  // VAE challenges
  { id: 'vae-ch-1', title: 'Latent Space Explorer', description: 'Explore the 2D latent space and generate new patterns', difficulty: 'easy', xp_reward: 150, badge_reward: 'vae-encoder', requirements: {} },
  { id: 'vae-ch-2', title: 'Pattern Morphing', description: 'Interpolate between a circle and a cross in VAE', difficulty: 'medium', xp_reward: 200, badge_reward: 'vae-interpolator', requirements: {} },
  // Transformer challenges
  { id: 'tf-ch-1', title: 'Attention Patterns', description: 'Analyze attention heatmaps for "Attention is all you need"', difficulty: 'easy', xp_reward: 150, badge_reward: 'tf-attention', requirements: {} },
  { id: 'tf-ch-2', title: 'Multi-Head Analysis', description: 'Compare attention patterns across multiple heads', difficulty: 'medium', xp_reward: 200, badge_reward: 'tf-multihead', requirements: {} },
  // Cross-topic
  { id: 'cross-ch-1', title: 'Complete Scholar', description: 'Complete all 6 lessons in any single topic', difficulty: 'medium', xp_reward: 250, badge_reward: 'scholar', requirements: {} },
  { id: 'cross-ch-2', title: 'Master All Topics', description: 'Complete at least one lesson in each of the 5 topics', difficulty: 'hard', xp_reward: 500, badge_reward: 'polymath', requirements: {} },
];
