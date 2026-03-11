import React from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { TOPICS } from '../types/topics';
import CNNLab from './CNNLab';

export const Lab: React.FC = () => {
  const { topicId } = useParams();
  const navigate = useNavigate();
  const currentTopic = TOPICS.find(t => t.id === topicId);

  if (topicId === 'cnn') {
    return <CNNLab />;
  }

  if (topicId !== 'cnn') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center">
        <Card className="max-w-md">
          <div className="p-8 text-center">
            <h2 className="text-2xl font-bold text-white mb-4">Coming Soon</h2>
            <p className="text-slate-300 mb-6">
              The {currentTopic?.name} lab is under development. Check back soon!
            </p>
            <Button onClick={() => navigate('/topics')}>
              Back to Topics
            </Button>
          </div>
        </Card>
      </div>
    );
  }
  return null;
};
