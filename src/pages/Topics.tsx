import { useNavigate } from 'react-router-dom';
import { Brain, Image, Activity, Sparkles, Zap } from 'lucide-react';
import { TOPICS, TopicId } from '../types/topics';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';

const iconMap = {
  Brain,
  Image,
  Activity,
  Sparkles,
  Zap
};

export function Topics() {
  const navigate = useNavigate();

  const handleTopicSelect = (topicId: TopicId) => {
    navigate(`/topics/${topicId}/lessons`);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      <nav className="border-b border-slate-700/50 bg-slate-900/50 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <button
              onClick={() => navigate('/')}
              className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent"
            >
              Neural Network Learn
            </button>
            <div className="flex gap-4">
              <Button variant="secondary" onClick={() => navigate('/dashboard')}>
                Dashboard
              </Button>
              <Button variant="secondary" onClick={() => navigate('/leaderboard')}>
                Leaderboard
              </Button>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
            Choose Your Learning Path
          </h1>
          <p className="text-xl text-slate-300">
            Master deep learning through interactive simulations and hands-on experimentation
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {TOPICS.map((topic) => {
            const Icon = iconMap[topic.icon as keyof typeof iconMap];
            return (
              <Card key={topic.id} className="hover:scale-105 transition-transform duration-200">
                <div className="p-6">
                  <div className={`w-16 h-16 rounded-2xl bg-gradient-to-br ${topic.gradient} flex items-center justify-center mb-4`}>
                    <Icon className="w-8 h-8 text-white" />
                  </div>
                  <h3 className="text-2xl font-bold text-white mb-2">{topic.name}</h3>
                  <p className="text-slate-300 mb-6 min-h-[60px]">{topic.description}</p>
                  <div className="flex gap-3">
                    <Button
                      onClick={() => handleTopicSelect(topic.id)}
                      className="flex-1"
                    >
                      Start Learning
                    </Button>
                    <Button
                      variant="secondary"
                      onClick={() => navigate(`/topics/${topic.id}/lab`)}
                      className="flex-1"
                    >
                      Lab
                    </Button>
                  </div>
                </div>
              </Card>
            );
          })}
        </div>

        <div className="mt-12 text-center">
          <Card className="inline-block">
            <div className="p-6">
              <h3 className="text-xl font-semibold text-white mb-2">
                Track Your Progress
              </h3>
              <p className="text-slate-300 mb-4">
                Complete lessons, earn XP, unlock badges, and compete on the leaderboard
              </p>
              <Button onClick={() => navigate('/dashboard')}>
                View Dashboard
              </Button>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
