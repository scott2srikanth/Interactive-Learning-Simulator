import React from 'react';
import { Link } from 'react-router-dom';
import { Brain, Layers, Zap, Target, BookOpen, Trophy } from 'lucide-react';
import { Button } from '../components/ui/Button';
import { motion } from 'framer-motion';

export const Landing: React.FC = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50">
      <nav className="bg-white/80 backdrop-blur-md border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-2">
              <Brain className="w-8 h-8 text-blue-600" />
              <span className="text-xl font-bold text-gray-900">Neural Network Learn</span>
            </div>
            <div className="flex space-x-4">
              <Link to="/topics">
                <Button variant="ghost">Topics</Button>
              </Link>
              <Link to="/dashboard">
                <Button variant="ghost">Dashboard</Button>
              </Link>
              <Link to="/leaderboard">
                <Button variant="primary">Get Started</Button>
              </Link>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center"
        >
          <h1 className="text-6xl font-bold text-gray-900 mb-6">
            Master <span className="text-blue-600">Deep Learning</span> Visually
          </h1>
          <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
            Build, visualize, and understand neural networks through interactive simulations. Learn ANN, CNN, RNN, VAE, and Transformers – all in your browser.
          </p>
          <div className="flex justify-center space-x-4">
            <Link to="/topics">
              <Button size="lg">
                <BookOpen className="w-5 h-5 mr-2" />
                Start Learning
              </Button>
            </Link>
            <Link to="/dashboard">
              <Button size="lg" variant="outline">
                <Trophy className="w-5 h-5 mr-2" />
                View Progress
              </Button>
            </Link>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-24"
        >
          <FeatureCard
            icon={<Layers className="w-12 h-12 text-blue-600" />}
            title="Interactive Builder"
            description="Drag and drop layers to build CNN architectures. See real-time visualizations of feature maps and activations."
          />
          <FeatureCard
            icon={<Zap className="w-12 h-12 text-green-600" />}
            title="Real-Time Simulation"
            description="Watch your network process images in real-time. Understand convolution, pooling, and activation functions visually."
          />
          <FeatureCard
            icon={<Target className="w-12 h-12 text-red-600" />}
            title="Hands-On Learning"
            description="Complete interactive lessons and challenges. Earn XP and badges as you master CNN concepts."
          />
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="mt-32 bg-white rounded-2xl shadow-2xl p-12"
        >
          <h2 className="text-4xl font-bold text-gray-900 mb-8 text-center">
            Why CNN Learn?
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
            <div>
              <h3 className="text-2xl font-semibold text-gray-800 mb-4">For Beginners</h3>
              <ul className="space-y-3 text-gray-600">
                <li className="flex items-start">
                  <span className="text-green-500 mr-2">✓</span>
                  No prior machine learning experience required
                </li>
                <li className="flex items-start">
                  <span className="text-green-500 mr-2">✓</span>
                  Visual explanations of complex concepts
                </li>
                <li className="flex items-start">
                  <span className="text-green-500 mr-2">✓</span>
                  Structured learning path from basics to advanced
                </li>
              </ul>
            </div>
            <div>
              <h3 className="text-2xl font-semibold text-gray-800 mb-4">For Practitioners</h3>
              <ul className="space-y-3 text-gray-600">
                <li className="flex items-start">
                  <span className="text-green-500 mr-2">✓</span>
                  Experiment with architectures instantly
                </li>
                <li className="flex items-start">
                  <span className="text-green-500 mr-2">✓</span>
                  Visualize kernel operations and feature maps
                </li>
                <li className="flex items-start">
                  <span className="text-green-500 mr-2">✓</span>
                  Understand parameter counts and model complexity
                </li>
              </ul>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="mt-24 text-center"
        >
          <Trophy className="w-16 h-16 text-yellow-500 mx-auto mb-6" />
          <h2 className="text-4xl font-bold text-gray-900 mb-4">Start Your Journey Today</h2>
          <p className="text-xl text-gray-600 mb-8">
            Join thousands of learners mastering CNNs through interactive visualization
          </p>
          <Link to="/lessons">
            <Button size="lg">
              Begin First Lesson
            </Button>
          </Link>
        </motion.div>
      </div>

      <footer className="bg-gray-900 text-white mt-32">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="text-center">
            <div className="flex items-center justify-center space-x-2 mb-4">
              <Brain className="w-6 h-6" />
              <span className="text-lg font-semibold">CNN Learn</span>
            </div>
            <p className="text-gray-400">
              Interactive CNN Learning Platform
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
};

interface FeatureCardProps {
  icon: React.ReactNode;
  title: string;
  description: string;
}

const FeatureCard: React.FC<FeatureCardProps> = ({ icon, title, description }) => {
  return (
    <motion.div
      whileHover={{ scale: 1.05 }}
      className="bg-white rounded-xl p-8 shadow-lg border border-gray-200 hover:shadow-xl transition-shadow"
    >
      <div className="mb-4">{icon}</div>
      <h3 className="text-xl font-semibold text-gray-900 mb-3">{title}</h3>
      <p className="text-gray-600">{description}</p>
    </motion.div>
  );
};
