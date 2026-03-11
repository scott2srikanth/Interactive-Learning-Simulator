import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { useCNNStore } from '../../store/cnnStore';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';
import { Play, TrendingUp, TrendingDown } from 'lucide-react';

export const TrainingDashboard: React.FC = () => {
  const { trainingMetrics, simulateTraining, getTotalParameters } = useCNNStore();

  const handleStartTraining = () => {
    simulateTraining(50);
  };

  const latestMetrics = trainingMetrics[trainingMetrics.length - 1];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-2xl font-bold text-gray-900">Training Simulation</h3>
        <Button onClick={handleStartTraining}>
          <Play className="w-4 h-4 mr-2" />
          Start Training
        </Button>
      </div>

      {trainingMetrics.length > 0 && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Final Accuracy</p>
                  <p className="text-3xl font-bold text-green-600">
                    {(latestMetrics.accuracy * 100).toFixed(1)}%
                  </p>
                </div>
                <TrendingUp className="w-8 h-8 text-green-600" />
              </div>
            </Card>

            <Card>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Final Loss</p>
                  <p className="text-3xl font-bold text-blue-600">
                    {latestMetrics.loss.toFixed(4)}
                  </p>
                </div>
                <TrendingDown className="w-8 h-8 text-blue-600" />
              </div>
            </Card>

            <Card>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Total Parameters</p>
                  <p className="text-3xl font-bold text-gray-900">
                    {getTotalParameters().toLocaleString()}
                  </p>
                </div>
              </div>
            </Card>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card title="Training Loss">
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={trainingMetrics}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
                  <YAxis label={{ value: 'Loss', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="loss" stroke="#3b82f6" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Training Accuracy">
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={trainingMetrics}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
                  <YAxis label={{ value: 'Accuracy', angle: -90, position: 'insideLeft' }} domain={[0, 1]} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="accuracy" stroke="#10b981" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          </div>
        </>
      )}

      {trainingMetrics.length === 0 && (
        <Card>
          <div className="text-center py-12">
            <Play className="w-16 h-16 text-gray-400 mx-auto mb-4" />
            <p className="text-lg text-gray-600">Click "Start Training" to simulate the training process</p>
          </div>
        </Card>
      )}
    </div>
  );
};
