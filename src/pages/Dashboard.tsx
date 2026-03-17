import React from 'react';
import { Link } from 'react-router-dom';
import { useUserStore, badges, challenges } from '../store/userStore';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Navbar, NavLink } from '../components/ui/Navbar';
import { Trophy, Star, Award, Target, BookOpen, Brain } from 'lucide-react';
import { motion } from 'framer-motion';
import { TOPICS } from '../types/topics';

const TOPIC_INFO: Record<string, { prefix: string; count: number; color: string; icon: string }> = {
  cnn: { prefix: 'cnn-', count: 6, color: 'green', icon: '🔍' },
  ann: { prefix: 'ann-', count: 6, color: 'blue', icon: '🧠' },
  rnn: { prefix: 'rnn-', count: 6, color: 'orange', icon: '🔄' },
  vae: { prefix: 'vae-', count: 6, color: 'pink', icon: '✨' },
  transformers: { prefix: 'tf-', count: 13, color: 'violet', icon: '⚡' },
};

export const Dashboard: React.FC = () => {
  const { xp, level, completedLessons, earnedBadges, completedChallenges } = useUserStore();
  const xpProgress = (xp % 100) / 100;
  const completedBadgeList = badges.filter(b => earnedBadges.includes(b.id));
  const lockedBadges = badges.filter(b => !earnedBadges.includes(b.id));
  const availCh = challenges.filter(c => !completedChallenges.includes(c.id));
  const doneCh = challenges.filter(c => completedChallenges.includes(c.id));

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900">
      <Navbar actions={<><NavLink to="/">Home</NavLink><NavLink to="/topics">Topics</NavLink><NavLink to="/leaderboard">Leaderboard</NavLink></>} />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          {[
            { label: 'Level', val: level, icon: <Star className="w-10 h-10 text-blue-600 dark:text-blue-400" />, color: 'blue' },
            { label: 'Lessons', val: `${completedLessons.length}/37`, icon: <BookOpen className="w-10 h-10 text-green-600 dark:text-green-400" />, color: 'green' },
            { label: 'Badges', val: `${earnedBadges.length}/${badges.length}`, icon: <Trophy className="w-10 h-10 text-yellow-500" />, color: 'yellow' },
            { label: 'Challenges', val: `${completedChallenges.length}/${challenges.length}`, icon: <Target className="w-10 h-10 text-purple-600 dark:text-purple-400" />, color: 'purple' },
          ].map((s, i) => (
            <motion.div key={i} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.1 }}>
              <Card>
                <div className="flex items-center justify-between">
                  <div><p className="text-sm text-gray-500 dark:text-gray-400">{s.label}</p><p className="text-3xl font-bold text-gray-900 dark:text-white">{s.val}</p></div>
                  {s.icon}
                </div>
                {s.label === 'Level' && <div className="mt-3"><div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mb-1"><span>XP: {xp}</span><span>{level * 100}</span></div><div className="w-full bg-gray-200 dark:bg-slate-700 rounded-full h-2"><div className="bg-blue-600 h-2 rounded-full transition-all" style={{ width: `${xpProgress * 100}%` }} /></div></div>}
              </Card>
            </motion.div>
          ))}
        </div>

        {/* Topic progress */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.35 }} className="mb-8">
          <Card title="Progress by Topic">
            <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
              {Object.entries(TOPIC_INFO).map(([tid, info]) => {
                const done = completedLessons.filter(l => l.startsWith(info.prefix)).length;
                const pct = (done / info.count) * 100;
                const topic = TOPICS.find(t => t.id === tid);
                const colors: Record<string, string> = { green: 'bg-green-500', blue: 'bg-blue-500', orange: 'bg-orange-500', pink: 'bg-pink-500', violet: 'bg-violet-500' };
                return (
                  <Link key={tid} to={`/topics/${tid}/lessons`}>
                    <div className="p-4 rounded-lg border border-gray-200 dark:border-slate-700 hover:shadow-md dark:hover:shadow-slate-900/30 transition-shadow bg-white dark:bg-slate-800/50 cursor-pointer">
                      <div className="flex items-center gap-2 mb-2"><span className="text-xl">{info.icon}</span><span className="text-sm font-semibold text-gray-900 dark:text-white">{topic?.name?.split(' ')[0] || tid.toUpperCase()}</span></div>
                      <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">{done}/{info.count} lessons</p>
                      <div className="w-full bg-gray-200 dark:bg-slate-700 rounded-full h-2"><div className={`h-2 rounded-full transition-all ${colors[info.color]}`} style={{ width: `${pct}%` }} /></div>
                      {pct === 100 && <p className="text-xs text-green-500 font-semibold mt-1">✓ Complete!</p>}
                    </div>
                  </Link>
                );
              })}
            </div>
          </Card>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Badges */}
          <Card title={`Badges (${completedBadgeList.length}/${badges.length})`}>
            <div className="space-y-3">
              {completedBadgeList.length > 0 ? completedBadgeList.map(b => (
                <div key={b.id} className="flex items-center gap-4 p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
                  <div className="text-3xl">{b.icon}</div>
                  <div className="flex-1"><h4 className="font-semibold text-gray-900 dark:text-white">{b.name}</h4><p className="text-xs text-gray-500 dark:text-gray-400">{b.description}</p></div>
                  <Award className="w-5 h-5 text-yellow-500" />
                </div>
              )) : <div className="text-center py-6 text-gray-400"><Trophy className="w-10 h-10 mx-auto mb-2 opacity-40" /><p className="text-sm">No badges yet</p></div>}
              {lockedBadges.length > 0 && <div className="mt-4"><p className="text-xs font-semibold text-gray-500 dark:text-gray-400 mb-2">Locked ({lockedBadges.length})</p><div className="space-y-1 max-h-40 overflow-y-auto">{lockedBadges.map(b => (
                <div key={b.id} className="flex items-center gap-3 p-2 bg-gray-50 dark:bg-slate-800 rounded opacity-50"><span className="text-xl grayscale">{b.icon}</span><div><p className="text-xs font-medium text-gray-700 dark:text-gray-300">{b.name}</p><p className="text-xs text-gray-400">{b.requirement}</p></div></div>
              ))}</div></div>}
            </div>
          </Card>

          {/* Challenges */}
          <Card title={`Challenges (${doneCh.length}/${challenges.length})`}>
            <div className="space-y-3">
              {availCh.slice(0, 5).map(c => {
                const tid = c.id.startsWith('cnn') ? 'cnn' : c.id.startsWith('ann') ? 'ann' : c.id.startsWith('rnn') ? 'rnn' : c.id.startsWith('vae') ? 'vae' : c.id.startsWith('tf') ? 'transformers' : '';
                return (
                  <div key={c.id} className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
                    <div className="flex items-center justify-between mb-1"><h4 className="font-semibold text-sm text-gray-900 dark:text-white">{c.title}</h4><span className={`text-xs px-2 py-0.5 rounded ${c.difficulty === 'easy' ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400' : c.difficulty === 'medium' ? 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400' : 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400'}`}>{c.difficulty}</span></div>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">{c.description}</p>
                    <div className="flex items-center justify-between"><span className="text-xs font-medium text-blue-600 dark:text-blue-400">+{c.xp_reward} XP</span><Link to={tid ? `/topics/${tid}/lab` : '/topics'}><Button size="sm">Start</Button></Link></div>
                  </div>
                );
              })}
              {availCh.length > 5 && <p className="text-xs text-gray-400 text-center">+{availCh.length - 5} more</p>}
            </div>
          </Card>
        </div>

        {/* Quick actions */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.6 }} className="mt-8">
          <Card title="Quick Actions">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Link to="/topics"><div className="p-6 bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg text-white hover:shadow-lg transition-shadow"><Brain className="w-8 h-8 mb-2" /><h4 className="font-semibold mb-1">Choose a Topic</h4><p className="text-sm text-blue-100">CNN, ANN, RNN, VAE, Transformers</p></div></Link>
              <Link to="/topics/cnn/lab"><div className="p-6 bg-gradient-to-br from-green-500 to-green-600 rounded-lg text-white hover:shadow-lg transition-shadow"><Target className="w-8 h-8 mb-2" /><h4 className="font-semibold mb-1">Open a Lab</h4><p className="text-sm text-green-100">Interactive simulators</p></div></Link>
              <Link to="/leaderboard"><div className="p-6 bg-gradient-to-br from-yellow-500 to-yellow-600 rounded-lg text-white hover:shadow-lg transition-shadow"><Trophy className="w-8 h-8 mb-2" /><h4 className="font-semibold mb-1">Leaderboard</h4><p className="text-sm text-yellow-100">See top learners</p></div></Link>
            </div>
          </Card>
        </motion.div>
      </div>
    </div>
  );
};
