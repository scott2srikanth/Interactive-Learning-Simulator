import React from 'react';
import { Link } from 'react-router-dom';
import { useUserStore, badges, challenges } from '../store/userStore';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Trophy, Star, Award, Target, BookOpen, Brain } from 'lucide-react';
import { motion } from 'framer-motion';

export const Dashboard: React.FC = () => {
  const { xp, level, completedLessons, earnedBadges, completedChallenges } = useUserStore();

  const xpForNextLevel = level * 100;
  const xpProgress = (xp % 100) / 100;

  const completedBadges = badges.filter((badge) => earnedBadges.includes(badge.id));
  const lockedBadges = badges.filter((badge) => !earnedBadges.includes(badge.id));

  const availableChallenges = challenges.filter((c) => !completedChallenges.includes(c.id));
  const completed = challenges.filter((c) => completedChallenges.includes(c.id));

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50">
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
            <div className="flex space-x-2">
              <Link to="/">
                <Button variant="ghost">Home</Button>
              </Link>
              <Link to="/lessons">
                <Button variant="ghost">Lessons</Button>
              </Link>
              <Link to="/lab">
                <Button variant="primary">Go to Lab</Button>
              </Link>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <Card>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Level</p>
                  <p className="text-4xl font-bold text-blue-600">{level}</p>
                </div>
                <Star className="w-12 h-12 text-blue-600" />
              </div>
              <div className="mt-4">
                <div className="flex justify-between text-sm text-gray-600 mb-1">
                  <span>XP: {xp}</span>
                  <span>{xpForNextLevel}</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full transition-all"
                    style={{ width: `${xpProgress * 100}%` }}
                  />
                </div>
              </div>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <Card>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Lessons Completed</p>
                  <p className="text-4xl font-bold text-green-600">{completedLessons.length}/6</p>
                </div>
                <BookOpen className="w-12 h-12 text-green-600" />
              </div>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <Card>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Badges Earned</p>
                  <p className="text-4xl font-bold text-yellow-600">{earnedBadges.length}/{badges.length}</p>
                </div>
                <Trophy className="w-12 h-12 text-yellow-600" />
              </div>
            </Card>
          </motion.div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <Card title="Your Badges">
              <div className="space-y-4">
                {completedBadges.length > 0 ? (
                  completedBadges.map((badge) => (
                    <div
                      key={badge.id}
                      className="flex items-center space-x-4 p-4 bg-yellow-50 rounded-lg border border-yellow-200"
                    >
                      <div className="text-4xl">{badge.icon}</div>
                      <div className="flex-1">
                        <h4 className="font-semibold text-gray-900">{badge.name}</h4>
                        <p className="text-sm text-gray-600">{badge.description}</p>
                      </div>
                      <Award className="w-6 h-6 text-yellow-600" />
                    </div>
                  ))
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    <Trophy className="w-12 h-12 mx-auto mb-2 text-gray-400" />
                    <p>No badges earned yet. Complete challenges to earn badges!</p>
                  </div>
                )}

                {lockedBadges.length > 0 && (
                  <div className="mt-6">
                    <h4 className="text-sm font-semibold text-gray-700 mb-3">Locked Badges</h4>
                    <div className="space-y-2">
                      {lockedBadges.map((badge) => (
                        <div
                          key={badge.id}
                          className="flex items-center space-x-4 p-3 bg-gray-50 rounded-lg opacity-60"
                        >
                          <div className="text-2xl grayscale">{badge.icon}</div>
                          <div className="flex-1">
                            <h5 className="font-medium text-gray-700">{badge.name}</h5>
                            <p className="text-xs text-gray-500">{badge.requirement}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            <Card title="Active Challenges">
              <div className="space-y-4">
                {availableChallenges.length > 0 ? (
                  availableChallenges.map((challenge) => (
                    <div
                      key={challenge.id}
                      className="p-4 bg-blue-50 rounded-lg border border-blue-200"
                    >
                      <div className="flex items-start justify-between mb-2">
                        <h4 className="font-semibold text-gray-900">{challenge.title}</h4>
                        <span
                          className={`text-xs px-2 py-1 rounded ${
                            challenge.difficulty === 'easy'
                              ? 'bg-green-100 text-green-800'
                              : challenge.difficulty === 'medium'
                              ? 'bg-yellow-100 text-yellow-800'
                              : 'bg-red-100 text-red-800'
                          }`}
                        >
                          {challenge.difficulty}
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 mb-3">{challenge.description}</p>
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium text-blue-600">
                          +{challenge.xp_reward} XP
                        </span>
                        <Link to="/lab">
                          <Button size="sm">Start Challenge</Button>
                        </Link>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    <Target className="w-12 h-12 mx-auto mb-2 text-gray-400" />
                    <p>All challenges completed!</p>
                  </div>
                )}

                {completed.length > 0 && (
                  <div className="mt-6">
                    <h4 className="text-sm font-semibold text-gray-700 mb-3">Completed</h4>
                    <div className="space-y-2">
                      {completed.map((challenge) => (
                        <div
                          key={challenge.id}
                          className="p-3 bg-green-50 rounded-lg border border-green-200 opacity-75"
                        >
                          <div className="flex items-center justify-between">
                            <span className="font-medium text-gray-700">{challenge.title}</span>
                            <span className="text-green-600 text-sm">✓ Completed</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </Card>
          </motion.div>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.5 }}
          className="mt-8"
        >
          <Card title="Quick Actions">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Link to="/lessons">
                <div className="p-6 bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg text-white hover:shadow-lg transition-shadow cursor-pointer">
                  <Brain className="w-8 h-8 mb-2" />
                  <h4 className="font-semibold mb-1">Continue Learning</h4>
                  <p className="text-sm text-blue-100">Resume your lessons</p>
                </div>
              </Link>
              <Link to="/lab">
                <div className="p-6 bg-gradient-to-br from-green-500 to-green-600 rounded-lg text-white hover:shadow-lg transition-shadow cursor-pointer">
                  <Target className="w-8 h-8 mb-2" />
                  <h4 className="font-semibold mb-1">Open Lab</h4>
                  <p className="text-sm text-green-100">Build your own CNN</p>
                </div>
              </Link>
              <Link to="/leaderboard">
                <div className="p-6 bg-gradient-to-br from-yellow-500 to-yellow-600 rounded-lg text-white hover:shadow-lg transition-shadow cursor-pointer">
                  <Trophy className="w-8 h-8 mb-2" />
                  <h4 className="font-semibold mb-1">Leaderboard</h4>
                  <p className="text-sm text-yellow-100">See top learners</p>
                </div>
              </Link>
            </div>
          </Card>
        </motion.div>
      </div>
    </div>
  );
};
