import { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Landing } from './pages/Landing';
import { Topics } from './pages/Topics';
import { Dashboard } from './pages/Dashboard';
import { Leaderboard } from './pages/Leaderboard';
import { Lab } from './pages/Lab';
import { Lessons } from './pages/Lessons';
import { LessonLab } from './pages/LessonLab';
import { useThemeStore } from './store/themeStore';

function App() {
  const dark = useThemeStore((s) => s.dark);

  useEffect(() => {
    if (dark) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [dark]);

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-slate-900 transition-colors">
      <Router>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/topics" element={<Topics />} />
          <Route path="/topics/:topicId/lab" element={<Lab />} />
          <Route path="/topics/:topicId/lab/:lessonId" element={<LessonLab />} />
          <Route path="/topics/:topicId/lessons" element={<Lessons />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/leaderboard" element={<Leaderboard />} />
        </Routes>
      </Router>
    </div>
  );
}

export default App;
