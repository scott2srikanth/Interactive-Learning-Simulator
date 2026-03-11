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
        const currentXP = get().xp;
        const newXP = currentXP + amount;
        const newLevel = Math.floor(newXP / 100) + 1;

        set({
          xp: newXP,
          level: newLevel,
        });
      },

      completeLesson: (lessonId) => {
        const { completedLessons } = get();
        if (!completedLessons.includes(lessonId)) {
          set({
            completedLessons: [...completedLessons, lessonId],
          });
          get().addXP(50);
        }
      },

      earnBadge: (badgeId) => {
        const { earnedBadges } = get();
        if (!earnedBadges.includes(badgeId)) {
          set({
            earnedBadges: [...earnedBadges, badgeId],
          });
          get().addXP(100);
        }
      },

      completeChallenge: (challengeId) => {
        const { completedChallenges } = get();
        if (!completedChallenges.includes(challengeId)) {
          set({
            completedChallenges: [...completedChallenges, challengeId],
          });
        }
      },

      setAuthenticated: (value) => set({ isAuthenticated: value }),

      resetProgress: () =>
        set({
          xp: 0,
          level: 1,
          completedLessons: [],
          earnedBadges: [],
          completedChallenges: [],
        }),
    }),
    {
      name: 'cnn-user-storage',
    }
  )
);

export const badges: Badge[] = [
  {
    id: 'first-network',
    name: 'First Network',
    description: 'Build your first CNN architecture',
    icon: '🌟',
    requirement: 'Create a CNN with at least 3 layers',
  },
  {
    id: 'edge-detector',
    name: 'Edge Detector',
    description: 'Successfully implement edge detection',
    icon: '🔍',
    requirement: 'Use edge detection kernel',
  },
  {
    id: 'efficiency-expert',
    name: 'Efficiency Expert',
    description: 'Build a CNN with less than 50k parameters',
    icon: '⚡',
    requirement: 'Parameters < 50,000',
  },
  {
    id: 'deep-learner',
    name: 'Deep Learner',
    description: 'Create a deep CNN with 10+ layers',
    icon: '🧠',
    requirement: 'Build CNN with 10+ layers',
  },
  {
    id: 'accuracy-master',
    name: 'Accuracy Master',
    description: 'Achieve 95%+ simulated accuracy',
    icon: '🎯',
    requirement: 'Simulated accuracy >= 95%',
  },
  {
    id: 'lesson-complete',
    name: 'Scholar',
    description: 'Complete all learning modules',
    icon: '📚',
    requirement: 'Complete all 6 lessons',
  },
];

export const challenges: Challenge[] = [
  {
    id: 'challenge-1',
    title: 'Efficient Architecture',
    description: 'Build a CNN with less than 50,000 parameters that achieves good accuracy',
    difficulty: 'easy',
    xp_reward: 150,
    badge_reward: 'efficiency-expert',
    requirements: {
      maxParameters: 50000,
    },
  },
  {
    id: 'challenge-2',
    title: 'High Accuracy',
    description: 'Create a CNN that achieves 95% simulated accuracy',
    difficulty: 'medium',
    xp_reward: 200,
    badge_reward: 'accuracy-master',
    requirements: {
      minAccuracy: 0.95,
    },
  },
  {
    id: 'challenge-3',
    title: 'Deep Network',
    description: 'Build a deep CNN with at least 10 layers',
    difficulty: 'hard',
    xp_reward: 300,
    badge_reward: 'deep-learner',
    requirements: {
      requiredLayers: ['conv2d', 'maxpool', 'dense'],
    },
  },
];
