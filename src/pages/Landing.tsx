import React from 'react';
import { Link } from 'react-router-dom';
import { Brain, Layers, Zap, Target, BookOpen, Trophy } from 'lucide-react';
import { Button } from '../components/ui/Button';
import { Navbar, NavLink } from '../components/ui/Navbar';
import { motion } from 'framer-motion';

export const Landing: React.FC = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900">
      <Navbar actions={<><NavLink to="/topics">Topics</NavLink><NavLink to="/dashboard">Dashboard</NavLink><NavLink to="/topics" primary>Get Started</NavLink></>} />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8 }} className="text-center">
          <h1 className="text-6xl font-bold text-gray-900 dark:text-white mb-6">
            Master <span className="text-blue-600 dark:text-blue-400">Deep Learning</span> Visually
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300 mb-8 max-w-3xl mx-auto">
            Build, visualize, and understand neural networks through interactive simulations. Learn ANN, CNN, RNN, VAE, and Transformers – all in your browser.
          </p>
          <div className="flex justify-center space-x-4">
            <Link to="/topics"><Button size="lg"><BookOpen className="w-5 h-5 mr-2" />Start Learning</Button></Link>
            <Link to="/dashboard"><Button size="lg" variant="outline"><Trophy className="w-5 h-5 mr-2" />View Progress</Button></Link>
          </div>
        </motion.div>

        <motion.div initial={{ opacity: 0, y: 40 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, delay: 0.2 }} className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-24">
          <FeatureCard icon={<Layers className="w-12 h-12 text-blue-600 dark:text-blue-400" />} title="Interactive Builder" description="Drag and drop layers to build architectures. See real-time visualizations of feature maps and activations." />
          <FeatureCard icon={<Zap className="w-12 h-12 text-green-600 dark:text-green-400" />} title="Real-Time Simulation" description="Watch your network process data in real-time. Understand convolution, attention, and more visually." />
          <FeatureCard icon={<Target className="w-12 h-12 text-red-600 dark:text-red-400" />} title="Hands-On Learning" description="Complete interactive lessons and challenges. Earn XP and badges as you master neural network concepts." />
        </motion.div>

        <motion.div initial={{ opacity: 0, y: 40 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, delay: 0.4 }} className="mt-32 bg-white dark:bg-slate-800 rounded-2xl shadow-2xl dark:shadow-slate-900/50 p-12 border border-gray-200 dark:border-slate-700">
          <h2 className="text-4xl font-bold text-gray-900 dark:text-white mb-8 text-center">Why Neural Network Learn?</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
            <div>
              <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">For Beginners</h3>
              <ul className="space-y-3 text-gray-600 dark:text-gray-400">
                <li className="flex items-start"><span className="text-green-500 mr-2">✓</span>No prior machine learning experience required</li>
                <li className="flex items-start"><span className="text-green-500 mr-2">✓</span>Visual explanations of complex concepts</li>
                <li className="flex items-start"><span className="text-green-500 mr-2">✓</span>Structured learning path from basics to advanced</li>
              </ul>
            </div>
            <div>
              <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">For Practitioners</h3>
              <ul className="space-y-3 text-gray-600 dark:text-gray-400">
                <li className="flex items-start"><span className="text-green-500 mr-2">✓</span>Experiment with architectures instantly</li>
                <li className="flex items-start"><span className="text-green-500 mr-2">✓</span>Visualize operations and feature maps</li>
                <li className="flex items-start"><span className="text-green-500 mr-2">✓</span>Understand parameter counts and complexity</li>
              </ul>
            </div>
          </div>
        </motion.div>

        <motion.div initial={{ opacity: 0, y: 40 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, delay: 0.6 }} className="mt-24 text-center">
          <Trophy className="w-16 h-16 text-yellow-500 mx-auto mb-6" />
          <h2 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">Start Your Journey Today</h2>
          <p className="text-xl text-gray-600 dark:text-gray-300 mb-8">Join learners mastering neural networks through interactive visualization</p>
          <Link to="/topics"><Button size="lg">Begin First Lesson</Button></Link>
        </motion.div>
      </div>

      <footer className="bg-gray-900 dark:bg-slate-950 text-white mt-32">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 text-center">
          <div className="flex items-center justify-center space-x-2 mb-4"><Brain className="w-6 h-6" /><span className="text-lg font-semibold">NN Learn</span></div>
          <p className="text-gray-400">Interactive Neural Network Learning Platform</p>
        </div>
      </footer>
    </div>
  );
};

const FeatureCard: React.FC<{ icon: React.ReactNode; title: string; description: string }> = ({ icon, title, description }) => (
  <motion.div whileHover={{ scale: 1.05 }} className="bg-white dark:bg-slate-800 rounded-xl p-8 shadow-lg border border-gray-200 dark:border-slate-700 hover:shadow-xl transition-shadow">
    <div className="mb-4">{icon}</div>
    <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">{title}</h3>
    <p className="text-gray-600 dark:text-gray-400">{description}</p>
  </motion.div>
);
