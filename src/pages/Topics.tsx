import { useNavigate } from 'react-router-dom';
import { Brain, Image, Activity, Sparkles, Zap } from 'lucide-react';
import { TOPICS, TopicId } from '../types/topics';
import { Button } from '../components/ui/Button';
import { Navbar, NavLink } from '../components/ui/Navbar';

const iconMap: Record<string, any> = { Brain, Image, Activity, Sparkles, Zap };

export function Topics() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900">
      <Navbar actions={<><NavLink to="/dashboard">Dashboard</NavLink><NavLink to="/leaderboard">Leaderboard</NavLink></>} />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold mb-4 text-gray-900 dark:text-white">
            Choose Your <span className="text-blue-600 dark:text-blue-400">Learning Path</span>
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300">
            Select a topic to start learning through interactive lessons and hands-on labs
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {TOPICS.map((topic) => {
            const IconComponent = iconMap[topic.icon] || Brain;
            return (
              <div key={topic.id} className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg dark:shadow-slate-900/50 border border-gray-200 dark:border-slate-700 overflow-hidden hover:shadow-xl transition-all hover:-translate-y-1 group">
                <div className="p-8">
                  <div className="mb-6">
                    <IconComponent className="w-14 h-14 text-blue-600 dark:text-blue-400 group-hover:scale-110 transition-transform" />
                  </div>
                  <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-3">{topic.name}</h3>
                  <p className="text-gray-600 dark:text-gray-400 mb-6 text-sm leading-relaxed">{topic.description}</p>
                  <div className="flex gap-3">
                    <Button size="sm" onClick={() => navigate(`/topics/${topic.id}/lessons`)}>
                      📚 Lessons
                    </Button>
                    <Button size="sm" variant="outline" onClick={() => navigate(`/topics/${topic.id}/lab`)}>
                      🧪 Lab
                    </Button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
