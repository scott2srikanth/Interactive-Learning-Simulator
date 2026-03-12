import { useThemeStore } from '../store/themeStore';

export interface LabColors {
  // Page
  pageBg: string;
  // Text
  text: string;
  textMuted: string;
  textDim: string;
  textWhite: string;
  // Surfaces
  cardBg: string;
  panelBg: string;
  inputBg: string;
  // Borders
  border: string;
  borderLight: string;
  // Nav bar
  navBg: string;
  navBorder: string;
  // Canvas
  canvasBg: string;
  canvasBorder: string;
  // Misc
  scrollThumb: string;
}

const DARK: LabColors = {
  pageBg: 'linear-gradient(145deg, #020617 0%, #0a1628 50%, #020617 100%)',
  text: '#e2e8f0',
  textMuted: '#94a3b8',
  textDim: '#64748b',
  textWhite: '#fff',
  cardBg: 'rgba(15,23,42,0.7)',
  panelBg: 'rgba(15,23,42,0.5)',
  inputBg: '#0f172a',
  border: '#1e293b',
  borderLight: '#334155',
  navBg: 'rgba(2,6,23,0.88)',
  navBorder: '#1e293b',
  canvasBg: '#0f172a',
  canvasBorder: '#334155',
  scrollThumb: '#334155',
};

const LIGHT: LabColors = {
  pageBg: 'linear-gradient(145deg, #f0f4ff 0%, #f8fafc 50%, #f0fdf4 100%)',
  text: '#1e293b',
  textMuted: '#475569',
  textDim: '#94a3b8',
  textWhite: '#0f172a',
  cardBg: 'rgba(255,255,255,0.85)',
  panelBg: 'rgba(241,245,249,0.8)',
  inputBg: '#f1f5f9',
  border: '#e2e8f0',
  borderLight: '#cbd5e1',
  navBg: 'rgba(255,255,255,0.92)',
  navBorder: '#e2e8f0',
  canvasBg: '#f8fafc',
  canvasBorder: '#cbd5e1',
  scrollThumb: '#cbd5e1',
};

export function useLabTheme(): LabColors {
  const dark = useThemeStore(s => s.dark);
  return dark ? DARK : LIGHT;
}

// Font constants shared by all labs
export const FONTS = {
  mono: "'IBM Plex Mono', monospace",
  sans: "'IBM Plex Sans', system-ui, sans-serif",
};
