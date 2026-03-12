import React from 'react';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  title?: string;
  subtitle?: string;
}

export const Card: React.FC<CardProps> = ({ children, className = '', title, subtitle }) => {
  return (
    <div className={`bg-white dark:bg-slate-800 rounded-xl shadow-lg border border-gray-200 dark:border-slate-700 ${className}`}>
      {(title || subtitle) && (
        <div className="p-6 border-b border-gray-200 dark:border-slate-700">
          {title && <h3 className="text-xl font-bold text-gray-900 dark:text-white">{title}</h3>}
          {subtitle && <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">{subtitle}</p>}
        </div>
      )}
      <div className="p-6">{children}</div>
    </div>
  );
};
