import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Brain, Sun, Moon } from 'lucide-react';
import { useThemeStore } from '../../store/themeStore';

interface NavbarProps {
  actions?: React.ReactNode;
}

export const Navbar: React.FC<NavbarProps> = ({ actions }) => {
  const { dark, toggle } = useThemeStore();
  const navigate = useNavigate();

  return (
    <nav className="sticky top-0 z-50 border-b border-gray-200 dark:border-slate-700/50 bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-14">
          <button onClick={() => navigate('/')} className="flex items-center space-x-2 group">
            <Brain className="w-7 h-7 text-blue-600 dark:text-blue-400 group-hover:scale-110 transition-transform" />
            <span className="text-lg font-bold text-gray-900 dark:text-white">NN Learn</span>
          </button>
          <div className="flex items-center gap-2">
            {actions}
            <button
              onClick={toggle}
              className="p-2 rounded-lg text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-slate-800 transition-colors"
              title={dark ? 'Switch to light mode' : 'Switch to dark mode'}
            >
              {dark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
};

export const NavLink: React.FC<{ to: string; children: React.ReactNode; primary?: boolean }> = ({ to, children, primary }) => (
  <Link
    to={to}
    className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
      primary
        ? 'bg-blue-600 text-white hover:bg-blue-700'
        : 'text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-slate-800'
    }`}
  >
    {children}
  </Link>
);
