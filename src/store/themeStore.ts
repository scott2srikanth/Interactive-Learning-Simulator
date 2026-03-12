import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface ThemeState {
  dark: boolean;
  toggle: () => void;
  set: (dark: boolean) => void;
}

export const useThemeStore = create<ThemeState>()(
  persist(
    (set) => ({
      dark: true,
      toggle: () => set((s) => ({ dark: !s.dark })),
      set: (dark) => set({ dark }),
    }),
    { name: 'nn-theme' }
  )
);
