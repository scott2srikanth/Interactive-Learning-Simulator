import React from 'react';
import { Link } from 'react-router-dom';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Navbar, NavLink } from '../components/ui/Navbar';
import { Trophy, Medal, Award } from 'lucide-react';
import { useUserStore } from '../store/userStore';

const mockLeaderboard = [
  { rank: 1, username: 'NeuralNinja', level: 15, xp: 1500, badges: 6 },
  { rank: 2, username: 'DeepDreamer', level: 12, xp: 1200, badges: 5 },
  { rank: 3, username: 'ConvKing', level: 10, xp: 1000, badges: 4 },
  { rank: 4, username: 'AIExplorer', level: 8, xp: 800, badges: 3 },
  { rank: 5, username: 'MLMaster', level: 7, xp: 700, badges: 3 },
];

export const Leaderboard: React.FC = () => {
  const { xp, level, earnedBadges } = useUserStore();
  const getRankIcon = (rank: number) => rank === 1 ? <Trophy className="w-6 h-6 text-yellow-500" /> : rank === 2 ? <Medal className="w-6 h-6 text-gray-400" /> : rank === 3 ? <Medal className="w-6 h-6 text-orange-500" /> : <span className="text-lg font-bold text-gray-500 dark:text-gray-400">#{rank}</span>;

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900">
      <Navbar actions={<><NavLink to="/">Home</NavLink><NavLink to="/dashboard">Dashboard</NavLink><NavLink to="/topics" primary>Topics</NavLink></>} />

      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Card>
          <div className="text-center mb-8">
            <Trophy className="w-16 h-16 text-yellow-500 mx-auto mb-4" />
            <h2 className="text-3xl font-bold text-gray-900 dark:text-white">Top Learners</h2>
            <p className="text-gray-600 dark:text-gray-400 mt-2">See how you rank against other learners</p>
          </div>

          {xp > 0 && (
            <div className="mb-6 p-4 bg-blue-50 dark:bg-blue-900/20 border-2 border-blue-300 dark:border-blue-700 rounded-lg">
              <div className="flex items-center justify-between flex-wrap gap-4">
                <div className="flex items-center gap-3"><Award className="w-6 h-6 text-blue-600 dark:text-blue-400" /><span className="font-semibold text-gray-900 dark:text-white">You</span></div>
                <div className="flex gap-6">
                  <div className="text-center"><p className="text-xs text-gray-500 dark:text-gray-400">Level</p><p className="text-lg font-bold text-gray-900 dark:text-white">{level}</p></div>
                  <div className="text-center"><p className="text-xs text-gray-500 dark:text-gray-400">XP</p><p className="text-lg font-bold text-blue-600 dark:text-blue-400">{xp}</p></div>
                  <div className="text-center"><p className="text-xs text-gray-500 dark:text-gray-400">Badges</p><p className="text-lg font-bold text-yellow-600 dark:text-yellow-400">{earnedBadges.length}</p></div>
                </div>
              </div>
            </div>
          )}

          <div className="space-y-2">
            {mockLeaderboard.map(e => (
              <div key={e.rank} className={`p-4 rounded-lg transition-all ${e.rank <= 3 ? 'bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/10 dark:to-orange-900/10 border-2 border-yellow-200 dark:border-yellow-800' : 'bg-gray-50 dark:bg-slate-800/50 hover:bg-gray-100 dark:hover:bg-slate-800'}`}>
                <div className="flex items-center justify-between flex-wrap gap-4">
                  <div className="flex items-center gap-4"><div className="w-10 flex justify-center">{getRankIcon(e.rank)}</div><span className="font-semibold text-gray-900 dark:text-white">{e.username}</span></div>
                  <div className="flex gap-6">
                    <div className="text-center"><p className="text-xs text-gray-500 dark:text-gray-400">Level</p><p className="text-lg font-bold text-gray-900 dark:text-white">{e.level}</p></div>
                    <div className="text-center"><p className="text-xs text-gray-500 dark:text-gray-400">XP</p><p className="text-lg font-bold text-blue-600 dark:text-blue-400">{e.xp}</p></div>
                    <div className="text-center"><p className="text-xs text-gray-500 dark:text-gray-400">Badges</p><p className="text-lg font-bold text-yellow-600 dark:text-yellow-400">{e.badges}</p></div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-8 text-center">
            <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">Keep learning to climb the leaderboard!</p>
            <div className="flex justify-center gap-4">
              <Link to="/topics"><Button variant="outline">Continue Learning</Button></Link>
              <Link to="/dashboard"><Button>View Progress</Button></Link>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
};
