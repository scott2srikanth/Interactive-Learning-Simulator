import React from 'react';
import { useParams, useNavigate, useLocation } from 'react-router-dom';
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

export const Lab: React.FC = () => {
  const { topicId } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const dark = useThemeStore(s => s.dark);
  const currentTopic = TOPICS.find(t => t.id === topicId);
  const LabComponent = LAB_MAP[topicId || ''];

  // Determine back navigation: came from lessons? go back there. Otherwise topics.
  const backTo = location.state?.from || `/topics/${topicId}/lessons`;
  const backLabel = location.state?.fromLabel || 'Lessons';

  if (!LabComponent) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900 flex items-center justify-center">
        <Card className="max-w-md">
          <div className="p-8 text-center">
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">Coming Soon</h2>
            <p className="text-gray-600 dark:text-gray-300 mb-6">
              The {currentTopic?.name} lab is under development.
            </p>
            <Button onClick={() => navigate('/topics')}>Back to Topics</Button>
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col">
      {/* Navbar overlays the lab */}
      <div className="relative z-50">
        <Navbar
          actions={
            <>
              <NavLink to={`/topics/${topicId}/lessons`}>📚 Lessons</NavLink>
              <NavLink to="/topics">Topics</NavLink>
              <NavLink to="/dashboard">Dashboard</NavLink>
            </>
          }
        />
      </div>

      {/* Lab content: in dark mode, labs render their own dark bg. In light mode, we apply CSS vars to override */}
      <div className="flex-1 relative">
        {!dark && (
          <style>{`
            .lab-light-override {
              --lab-bg: #f8fafc !important;
              --lab-text: #1e293b !important;
              --lab-muted: #64748b !important;
              --lab-border: #e2e8f0 !important;
              --lab-card: #ffffff !important;
              --lab-card-border: #e2e8f0 !important;
              --lab-input-bg: #f1f5f9 !important;
              --lab-input-border: #cbd5e1 !important;
              --lab-input-text: #1e293b !important;
            }
            .lab-light-override,
            .lab-light-override > div {
              background: linear-gradient(145deg, #f0f4ff 0%, #f8fafc 50%, #f0fdf4 100%) !important;
              color: #1e293b !important;
            }
            .lab-light-override h1,
            .lab-light-override h2,
            .lab-light-override h3,
            .lab-light-override h5,
            .lab-light-override b,
            .lab-light-override strong { color: #0f172a !important; }
            .lab-light-override p,
            .lab-light-override span,
            .lab-light-override label { color: #475569 !important; }
            .lab-light-override p b,
            .lab-light-override p strong,
            .lab-light-override span b { color: #1e293b !important; }
            /* Keep colored text intact */
            .lab-light-override [style*="color: #3b82f6"],
            .lab-light-override [style*="color: #22c55e"],
            .lab-light-override [style*="color: #ef4444"],
            .lab-light-override [style*="color: #f59e0b"],
            .lab-light-override [style*="color: #a855f7"],
            .lab-light-override [style*="color: #ec4899"],
            .lab-light-override [style*="color: #06b6d4"] { color: inherit !important; }
            /* Navbar override cards */
            .lab-light-override nav,
            .lab-light-override [style*="backdrop-filter"] {
              background: rgba(255,255,255,0.9) !important;
              border-color: #e2e8f0 !important;
            }
            .lab-light-override nav span,
            .lab-light-override nav h1 { color: #0f172a !important; }
            /* Panels and cards */
            .lab-light-override [style*="rgba(15,23,42"],
            .lab-light-override [style*="#0f172a"],
            .lab-light-override [style*="rgba(2,6,23"] {
              background: #ffffff !important;
              border-color: #e2e8f0 !important;
            }
            /* Select/input */
            .lab-light-override select,
            .lab-light-override input[type="number"] {
              background: #f1f5f9 !important;
              color: #1e293b !important;
              border-color: #cbd5e1 !important;
            }
            /* Buttons */
            .lab-light-override button {
              border-color: #cbd5e1 !important;
            }
            .lab-light-override button[style*="#1e293b"] {
              background: #f1f5f9 !important;
              color: #475569 !important;
            }
            /* Keep canvas and colored elements */
            .lab-light-override canvas { /* untouched */ }
            /* Scrollbar */
            .lab-light-override *::-webkit-scrollbar-thumb { background: #cbd5e1 !important; }
          `}</style>
        )}
        <div className={!dark ? 'lab-light-override' : ''}>
          <LabComponent />
        </div>
      </div>
    </div>
  );
};
