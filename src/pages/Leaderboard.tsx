import React from 'react';
import { Link } from 'react-router-dom';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Trophy, Medal, Award } from 'lucide-react';
import { useUserStore } from '../store/userStore';

interface LeaderboardEntry {
  rank: number;
  username: string;
  level: number;
  xp: number;
  badges: number;
}

const mockLeaderboard: LeaderboardEntry[] = [
  { rank: 1, username: 'NeuralNinja', level: 15, xp: 1500, badges: 6 },
  { rank: 2, username: 'DeepDreamer', level: 12, xp: 1200, badges: 5 },
  { rank: 3, username: 'ConvKing', level: 10, xp: 1000, badges: 4 },
  { rank: 4, username: 'AIExplorer', level: 8, xp: 800, badges: 3 },
  { rank: 5, username: 'MLMaster', level: 7, xp: 700, badges: 3 },
];

export const Leaderboard: React.FC = () => {
  const { xp, level, earnedBadges } = useUserStore();

  const currentUserRank: LeaderboardEntry = {
    rank: 0,
    username: 'You',
    level,
    xp,
    badges: earnedBadges.length,
  };

  const getRankIcon = (rank: number) => {
    switch (rank) {
      case 1:
        return <Trophy className="w-6 h-6 text-yellow-500" />;
      case 2:
        return <Medal className="w-6 h-6 text-gray-400" />;
      case 3:
        return <Medal className="w-6 h-6 text-orange-600" />;
      default:
        return <span className="text-lg font-bold text-gray-600">#{rank}</span>;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50">
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <h1 className="text-2xl font-bold text-gray-900">Leaderboard</h1>
            <div className="flex space-x-2">
              <Link to="/">
                <Button variant="ghost">Home</Button>
              </Link>
              <Link to="/dashboard">
                <Button variant="ghost">Dashboard</Button>
              </Link>
              <Link to="/topics">
                <Button variant="primary">Go to Lab</Button>
              </Link>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Card>
          <div className="text-center mb-8">
            <Trophy className="w-16 h-16 text-yellow-500 mx-auto mb-4" />
            <h2 className="text-3xl font-bold text-gray-900">Top Learners</h2>
            <p className="text-gray-600 mt-2">See how you rank against other CNN enthusiasts</p>
          </div>

          {currentUserRank.xp > 0 && (
            <div className="mb-6 p-4 bg-blue-50 border-2 border-blue-300 rounded-lg">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <Award className="w-6 h-6 text-blue-600" />
                  <div>
                    <span className="font-semibold text-gray-900">{currentUserRank.username}</span>
                    <span className="text-sm text-gray-600 ml-2">(You)</span>
                  </div>
                </div>
                <div className="flex items-center space-x-6">
                  <div className="text-center">
                    <p className="text-sm text-gray-600">Level</p>
                    <p className="text-lg font-bold text-gray-900">{currentUserRank.level}</p>
                  </div>
                  <div className="text-center">
                    <p className="text-sm text-gray-600">XP</p>
                    <p className="text-lg font-bold text-blue-600">{currentUserRank.xp}</p>
                  </div>
                  <div className="text-center">
                    <p className="text-sm text-gray-600">Badges</p>
                    <p className="text-lg font-bold text-yellow-600">{currentUserRank.badges}</p>
                  </div>
                </div>
              </div>
            </div>
          )}

          <div className="space-y-2">
            {mockLeaderboard.map((entry) => (
              <div
                key={entry.rank}
                className={`p-4 rounded-lg transition-all ${
                  entry.rank <= 3
                    ? 'bg-gradient-to-r from-yellow-50 to-orange-50 border-2 border-yellow-200'
                    : 'bg-gray-50 hover:bg-gray-100'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <div className="w-12 flex justify-center">{getRankIcon(entry.rank)}</div>
                    <div>
                      <span className="font-semibold text-gray-900">{entry.username}</span>
                    </div>
                  </div>
                  <div className="flex items-center space-x-6">
                    <div className="text-center">
                      <p className="text-sm text-gray-600">Level</p>
                      <p className="text-lg font-bold text-gray-900">{entry.level}</p>
                    </div>
                    <div className="text-center">
                      <p className="text-sm text-gray-600">XP</p>
                      <p className="text-lg font-bold text-blue-600">{entry.xp}</p>
                    </div>
                    <div className="text-center">
                      <p className="text-sm text-gray-600">Badges</p>
                      <p className="text-lg font-bold text-yellow-600">{entry.badges}</p>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-8 text-center">
            <p className="text-sm text-gray-600 mb-4">
              Keep learning and completing challenges to climb the leaderboard!
            </p>
            <div className="flex justify-center space-x-4">
              <Link to="/topics">
                <Button variant="outline">Continue Learning</Button>
              </Link>
              <Link to="/dashboard">
                <Button>View Your Progress</Button>
              </Link>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
};
