import React from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Navbar, NavLink } from '../components/ui/Navbar';
import { TOPICS } from '../types/topics';
import { useThemeStore } from '../store/themeStore';
import CNNLab from './CNNLab';
import ANNLab from './ANNLab';
import RNNLab from './RNNLab';
import VAELab from './VAELab';
import TransformerLab from './TransformerLab';

const LAB_MAP: Record<string, React.FC> = {
  cnn: CNNLab, ann: ANNLab, rnn: RNNLab, vae: VAELab, transformers: TransformerLab,
};

/* ═══════════════════════════════════════════════════════════
   LIGHT THEME CSS — comprehensive override for all lab inline styles
   Every lab uses inline styles with hardcoded dark colors.
   This CSS targets every pattern: backgrounds, text, borders, inputs.
   ═══════════════════════════════════════════════════════════ */
const LIGHT_CSS = `
.lab-light * { transition: none !important; }

/* ── Page-level backgrounds ── */
.lab-light,
.lab-light > div,
.lab-light > div > div {
  color: #1e293b !important;
}
.lab-light > div[style*="background"] {
  background: linear-gradient(145deg, #eff6ff, #f8fafc, #f0fdf4) !important;
}

/* ── All text: default to dark ── */
.lab-light h1, .lab-light h2, .lab-light h3, .lab-light h5,
.lab-light b, .lab-light strong {
  color: #0f172a !important;
}
.lab-light p, .lab-light label, .lab-light span {
  color: #334155 !important;
}

/* ── Preserve semantic colored text (blue, green, red, yellow, purple, pink, cyan) ── */
.lab-light [style*="color: #3b82f6"] { color: #2563eb !important; }
.lab-light [style*="color: #60a5fa"] { color: #2563eb !important; }
.lab-light [style*="color: #22c55e"] { color: #16a34a !important; }
.lab-light [style*="color: #ef4444"] { color: #dc2626 !important; }
.lab-light [style*="color: #f59e0b"] { color: #d97706 !important; }
.lab-light [style*="color: #facc15"] { color: #ca8a04 !important; }
.lab-light [style*="color: #a855f7"] { color: #7c3aed !important; }
.lab-light [style*="color: #c084fc"] { color: #7c3aed !important; }
.lab-light [style*="color: #ec4899"] { color: #db2777 !important; }
.lab-light [style*="color: #06b6d4"] { color: #0891b2 !important; }
/* White text on colored backgrounds should stay white */
.lab-light [style*="background: #3b82f6"] span,
.lab-light [style*="background: #16a34a"] span,
.lab-light [style*="background: #dc2626"] span,
.lab-light [style*="background: linear-gradient"] span,
.lab-light [style*="background: linear-gradient"] h1 { color: #fff !important; }
.lab-light button[style*="background: #16a34a"],
.lab-light button[style*="background: #dc2626"] { color: #fff !important; }
.lab-light button[style*="background: #3b82f6"] { color: #fff !important; }

/* ── Top bar / sticky nav inside labs ── */
.lab-light [style*="position: sticky"],
.lab-light [style*="backdrop-filter"],
.lab-light div[style*="rgba(2,6,23"] {
  background: rgba(255,255,255,0.95) !important;
  border-color: #e2e8f0 !important;
}
.lab-light [style*="position: sticky"] h1,
.lab-light [style*="position: sticky"] span {
  color: #0f172a !important;
}

/* ── Cards and panels ── */
.lab-light div[style*="rgba(15,23,42"] {
  background: #ffffff !important;
  border-color: #e2e8f0 !important;
}
.lab-light div[style*="background: #0f172a"] {
  background: #f1f5f9 !important;
  border-color: #e2e8f0 !important;
}

/* ── Sidebar panels ── */
.lab-light div[style*="border: 1px solid #1e293b"],
.lab-light div[style*="border: 1px solid #334155"] {
  border-color: #e2e8f0 !important;
}

/* ── Inputs, selects, number inputs ── */
.lab-light select,
.lab-light input[type="number"],
.lab-light input[type="text"] {
  background: #f1f5f9 !important;
  color: #1e293b !important;
  border-color: #cbd5e1 !important;
}

/* ── Range inputs ── */
.lab-light input[type="range"] {
  accent-color: #3b82f6;
}

/* ── Buttons with dark bg ── */
.lab-light button[style*="background: #1e293b"],
.lab-light button[style*="background:#1e293b"] {
  background: #e2e8f0 !important;
  color: #475569 !important;
  border-color: #cbd5e1 !important;
}
.lab-light button[style*="background: #0f172a"] {
  background: #f1f5f9 !important;
  color: #334155 !important;
  border-color: #e2e8f0 !important;
}

/* ── Layer cards (ANN/CNN) — the main visible issue ── */
.lab-light div[style*="borderRadius: 10"][style*="background: rgba(15"] {
  background: #ffffff !important;
  border-color: #e2e8f0 !important;
}
.lab-light div[style*="borderRadius: 10"][style*="border: 1px solid"] {
  border-color: #e2e8f0 !important;
}
.lab-light div[style*="borderTop: \"1px solid"] {
  border-color: #e2e8f0 !important;
}

/* ── Semi-transparent colored backgrounds (keep but lighten) ── */
.lab-light div[style*="background: #3b82f611"],
.lab-light div[style*="background: rgba(59,130,246,0.0"] {
  background: rgba(59,130,246,0.08) !important;
}
.lab-light div[style*="background: #22c55e22"] { background: rgba(34,197,94,0.08) !important; }
.lab-light div[style*="background: #a855f722"] { background: rgba(168,85,247,0.08) !important; }
.lab-light div[style*="background: #f59e0b18"],
.lab-light div[style*="background: #f59e0b22"] { background: rgba(245,158,11,0.08) !important; }
.lab-light div[style*="background: #ec489922"] { background: rgba(236,72,153,0.08) !important; }

/* ── Network canvas bg — keep dark for contrast on the network diagram ── */
.lab-light canvas {
  /* canvases keep their own rendering */
}

/* ── Colored neuron circles (ANN) — keep their colors ── */
.lab-light div[style*="borderRadius: 26"],
.lab-light div[style*="borderRadius: 12"],
.lab-light div[style*="border-radius: 50%"] {
  /* keep colored circles */
}

/* ── Scrollbar ── */
.lab-light *::-webkit-scrollbar-thumb {
  background: #cbd5e1 !important;
}
.lab-light *::-webkit-scrollbar-track {
  background: transparent !important;
}

/* ── Code/mono blocks ── */
.lab-light div[style*="fontFamily: 'IBM Plex Mono'"] p,
.lab-light div[style*="fontFamily: 'IBM Plex Mono'"] span {
  /* preserve mono styling */
}

/* ── Fullscreen modal ── */
.lab-light div[style*="position: fixed"][style*="inset: 0"] {
  background: rgba(248,250,252,0.97) !important;
}
.lab-light div[style*="position: fixed"] div[style*="background: rgba(15"] {
  background: #ffffff !important;
  border-color: #e2e8f0 !important;
}

/* ── Grid cells inside layer details keep their computed colors ── */

/* ── Make sure gradient text on header stays visible ── */
.lab-light div[style*="background: linear-gradient(135deg"] {
  /* keep gradient icons */
}
.lab-light div[style*="background: linear-gradient(135deg"] + h1,
.lab-light div[style*="background: linear-gradient(135deg"] ~ span {
  color: #0f172a !important;
}

/* ── Google font link ── */
.lab-light link { display: none; }
`;

export const Lab: React.FC = () => {
  const { topicId } = useParams();
  const navigate = useNavigate();
  const dark = useThemeStore(s => s.dark);
  const currentTopic = TOPICS.find(t => t.id === topicId);
  const LabComponent = LAB_MAP[topicId || ''];

  if (!LabComponent) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900 flex items-center justify-center">
        <Card className="max-w-md">
          <div className="p-8 text-center">
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">Coming Soon</h2>
            <p className="text-gray-600 dark:text-gray-300 mb-6">The {currentTopic?.name} lab is under development.</p>
            <Button onClick={() => navigate('/topics')}>Back to Topics</Button>
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col">
      {/* Shared navbar */}
      <div className="relative z-50">
        <Navbar actions={<>
          <NavLink to={`/topics/${topicId}/lessons`}>📚 Lessons</NavLink>
          <NavLink to="/topics">Topics</NavLink>
          <NavLink to="/dashboard">Dashboard</NavLink>
        </>} />
      </div>

      {/* Lab with theme override */}
      {!dark && <style>{LIGHT_CSS}</style>}
      <div className={`flex-1 ${!dark ? 'lab-light' : ''}`}>
        <LabComponent />
      </div>
    </div>
  );
};
