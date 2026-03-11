import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Landing } from './pages/Landing';
import { Topics } from './pages/Topics';
import { Dashboard } from './pages/Dashboard';
import { Leaderboard } from './pages/Leaderboard';
import { Lab } from './pages/Lab';
import { Lessons } from './pages/Lessons';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/topics" element={<Topics />} />
        <Route path="/topics/:topicId/lab" element={<Lab />} />
        <Route path="/topics/:topicId/lessons" element={<Lessons />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/leaderboard" element={<Leaderboard />} />
      </Routes>
    </Router>
  );
}

export default App;
