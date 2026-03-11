import React, { useState } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { useUserStore } from '../store/userStore';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { CheckCircle, Circle, Brain, Image, Grid3x3, Filter, Activity, Layers as LayersIcon, ArrowLeft } from 'lucide-react';
import { motion } from 'framer-motion';
import { FeatureMapGrid } from '../components/visualization/FeatureMapGrid';
import { createSampleImage } from '../lib/datasets';
import { convolve2D, createEdgeDetectionKernel, createBlurKernel } from '../lib/cnn/convolution';
import { TOPICS } from '../types/topics';

interface Lesson {
  id: string;
  title: string;
  description: string;
  icon: React.ReactNode;
  content: React.ReactNode;
}

const lessons: Lesson[] = [
  {
    id: 'lesson-1',
    title: 'Images as Data',
    description: 'Learn how computers see images as matrices of numbers',
    icon: <Image className="w-6 h-6" />,
    content: (
      <div className="space-y-6">
        <p className="text-gray-700">
          Images are represented as grids of numbers called pixels. Each pixel contains color information.
          For grayscale images, each pixel is a single number between 0 (black) and 1 (white).
        </p>
        <div className="bg-blue-50 p-6 rounded-lg">
          <h4 className="font-semibold mb-4">Example: A Simple Image</h4>
          <FeatureMapGrid data={createSampleImage('vertical')} />
          <p className="mt-4 text-sm text-gray-600">
            This vertical line is represented as a 28x28 matrix where white pixels = 1 and black pixels = 0.
          </p>
        </div>
      </div>
    ),
  },
  {
    id: 'lesson-2',
    title: 'Convolution Intuition',
    description: 'Understand how convolution detects patterns in images',
    icon: <Grid3x3 className="w-6 h-6" />,
    content: (
      <div className="space-y-6">
        <p className="text-gray-700">
          Convolution is a sliding window operation. We slide a small matrix (called a kernel or filter)
          across the image, multiplying overlapping values and summing them up.
        </p>
        <div className="bg-green-50 p-6 rounded-lg">
          <h4 className="font-semibold mb-4">How It Works</h4>
          <ol className="list-decimal list-inside space-y-2 text-gray-700">
            <li>Place the kernel on the top-left of the image</li>
            <li>Multiply each kernel value with the corresponding image pixel</li>
            <li>Sum all the products to get a single output value</li>
            <li>Slide the kernel one step and repeat</li>
          </ol>
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <h5 className="font-semibold mb-2">Input Image</h5>
            <FeatureMapGrid data={createSampleImage('circle')} />
          </div>
          <div>
            <h5 className="font-semibold mb-2">After Convolution</h5>
            <FeatureMapGrid
              data={convolve2D(createSampleImage('circle')[0], createEdgeDetectionKernel(), 1)}
            />
          </div>
        </div>
      </div>
    ),
  },
  {
    id: 'lesson-3',
    title: 'Filters and Kernels',
    description: 'Discover how different kernels detect different features',
    icon: <Filter className="w-6 h-6" />,
    content: (
      <div className="space-y-6">
        <p className="text-gray-700">
          Different kernels detect different features. Edge detection kernels find edges, blur kernels smooth images,
          and sharpen kernels enhance details.
        </p>
        <div className="space-y-4">
          <div className="bg-yellow-50 p-6 rounded-lg">
            <h4 className="font-semibold mb-4">Edge Detection</h4>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm mb-2">Original</p>
                <FeatureMapGrid data={createSampleImage('diagonal')} />
              </div>
              <div>
                <p className="text-sm mb-2">Edges Detected</p>
                <FeatureMapGrid
                  data={convolve2D(createSampleImage('diagonal')[0], createEdgeDetectionKernel(), 1)}
                />
              </div>
            </div>
          </div>

          <div className="bg-blue-50 p-6 rounded-lg">
            <h4 className="font-semibold mb-4">Blur Filter</h4>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm mb-2">Original</p>
                <FeatureMapGrid data={createSampleImage('circle')} />
              </div>
              <div>
                <p className="text-sm mb-2">Blurred</p>
                <FeatureMapGrid
                  data={convolve2D(createSampleImage('circle')[0], createBlurKernel(), 1)}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    ),
  },
  {
    id: 'lesson-4',
    title: 'Activation Functions',
    description: 'Learn how activation functions introduce non-linearity',
    icon: <Activity className="w-6 h-6" />,
    content: (
      <div className="space-y-6">
        <p className="text-gray-700">
          Activation functions decide which neurons should be activated. They introduce non-linearity,
          allowing neural networks to learn complex patterns.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-red-50 p-4 rounded-lg">
            <h4 className="font-semibold mb-2">ReLU</h4>
            <p className="text-sm text-gray-700 mb-2">f(x) = max(0, x)</p>
            <p className="text-xs text-gray-600">Sets negative values to zero. Most commonly used.</p>
          </div>
          <div className="bg-green-50 p-4 rounded-lg">
            <h4 className="font-semibold mb-2">Sigmoid</h4>
            <p className="text-sm text-gray-700 mb-2">f(x) = 1/(1+e^-x)</p>
            <p className="text-xs text-gray-600">Squashes values between 0 and 1.</p>
          </div>
          <div className="bg-blue-50 p-4 rounded-lg">
            <h4 className="font-semibold mb-2">Tanh</h4>
            <p className="text-sm text-gray-700 mb-2">f(x) = tanh(x)</p>
            <p className="text-xs text-gray-600">Squashes values between -1 and 1.</p>
          </div>
        </div>
      </div>
    ),
  },
  {
    id: 'lesson-5',
    title: 'Pooling Layers',
    description: 'Understand how pooling reduces spatial dimensions',
    icon: <Grid3x3 className="w-6 h-6" />,
    content: (
      <div className="space-y-6">
        <p className="text-gray-700">
          Pooling layers reduce the spatial size of feature maps, making the network more efficient
          and helping it focus on the most important features.
        </p>
        <div className="space-y-4">
          <div className="bg-orange-50 p-6 rounded-lg">
            <h4 className="font-semibold mb-2">Max Pooling</h4>
            <p className="text-sm text-gray-700">
              Takes the maximum value from each region. Commonly uses a 2x2 window with stride 2,
              which reduces the image size by half in each dimension.
            </p>
          </div>
          <div className="bg-teal-50 p-6 rounded-lg">
            <h4 className="font-semibold mb-2">Average Pooling</h4>
            <p className="text-sm text-gray-700">
              Takes the average value from each region. More gentle than max pooling but less commonly used.
            </p>
          </div>
        </div>
      </div>
    ),
  },
  {
    id: 'lesson-6',
    title: 'CNN Architecture',
    description: 'Put it all together to build a complete CNN',
    icon: <LayersIcon className="w-6 h-6" />,
    content: (
      <div className="space-y-6">
        <p className="text-gray-700">
          A typical CNN architecture combines multiple layers in sequence to learn hierarchical features.
        </p>
        <div className="bg-gradient-to-r from-blue-50 to-green-50 p-6 rounded-lg">
          <h4 className="font-semibold mb-4">Typical CNN Flow</h4>
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 rounded-full bg-green-500"></div>
              <span>Input Image (28x28)</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 rounded-full bg-blue-500"></div>
              <span>Conv Layer (32 filters) → Feature extraction</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 rounded-full bg-orange-500"></div>
              <span>Max Pooling → Dimensionality reduction</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 rounded-full bg-blue-500"></div>
              <span>Conv Layer (64 filters) → More features</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 rounded-full bg-orange-500"></div>
              <span>Max Pooling → Further reduction</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 rounded-full bg-purple-500"></div>
              <span>Flatten → Convert to 1D</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 rounded-full bg-pink-500"></div>
              <span>Dense Layer → Classification</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 rounded-full bg-red-500"></div>
              <span>Softmax → Probabilities</span>
            </div>
          </div>
        </div>
        <div className="bg-yellow-50 p-6 rounded-lg">
          <h4 className="font-semibold mb-2">Ready to Build Your Own?</h4>
          <p className="text-sm text-gray-700 mb-4">
            Head to the Lab to create and experiment with your own CNN architectures!
          </p>
          <Link to="/topics/cnn/lab">
            <Button>Open Lab</Button>
          </Link>
        </div>
      </div>
    ),
  },
];

export const Lessons: React.FC = () => {
  const { topicId } = useParams();
  const navigate = useNavigate();
  const currentTopic = TOPICS.find(t => t.id === topicId);
  const { completedLessons, completeLesson } = useUserStore();
  const [selectedLesson, setSelectedLesson] = useState<Lesson | null>(null);

  if (topicId !== 'cnn') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center">
        <Card className="max-w-md">
          <div className="p-8 text-center">
            <h2 className="text-2xl font-bold text-white mb-4">Coming Soon</h2>
            <p className="text-slate-300 mb-6">
              The {currentTopic?.name} lessons are under development. Check back soon!
            </p>
            <Button onClick={() => navigate('/topics')}>
              Back to Topics
            </Button>
          </div>
        </Card>
      </div>
    );
  }

  const handleCompleteLesson = (lessonId: string) => {
    completeLesson(lessonId);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50">
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-4">
              <Button
                variant="ghost"
                onClick={() => navigate('/topics')}
              >
                <ArrowLeft className="w-4 h-4 mr-2" />
                Topics
              </Button>
              <h1 className="text-2xl font-bold text-gray-900">CNN Lessons</h1>
            </div>
            <div className="flex space-x-2">
              <Link to="/dashboard">
                <Button variant="ghost">Dashboard</Button>
              </Link>
              <Link to="/topics/cnn/lab">
                <Button variant="primary">Go to Lab</Button>
              </Link>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {!selectedLesson ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {lessons.map((lesson, idx) => {
              const isCompleted = completedLessons.includes(lesson.id);

              return (
                <motion.div
                  key={lesson.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: idx * 0.1 }}
                >
                  <Card className="h-full hover:shadow-xl transition-shadow cursor-pointer">
                    <div onClick={() => setSelectedLesson(lesson)}>
                      <div className="flex items-start justify-between mb-4">
                        <div className="p-3 bg-blue-100 rounded-lg">{lesson.icon}</div>
                        {isCompleted ? (
                          <CheckCircle className="w-6 h-6 text-green-600" />
                        ) : (
                          <Circle className="w-6 h-6 text-gray-300" />
                        )}
                      </div>
                      <h3 className="text-xl font-semibold text-gray-900 mb-2">{lesson.title}</h3>
                      <p className="text-gray-600 text-sm mb-4">{lesson.description}</p>
                      <Button variant="outline" size="sm" className="w-full">
                        {isCompleted ? 'Review Lesson' : 'Start Lesson'}
                      </Button>
                    </div>
                  </Card>
                </motion.div>
              );
            })}
          </div>
        ) : (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
          >
            <Card>
              <div className="mb-6">
                <Button variant="ghost" onClick={() => setSelectedLesson(null)}>
                  ← Back to Lessons
                </Button>
              </div>
              <div className="flex items-center space-x-4 mb-6">
                <div className="p-3 bg-blue-100 rounded-lg">{selectedLesson.icon}</div>
                <div>
                  <h2 className="text-3xl font-bold text-gray-900">{selectedLesson.title}</h2>
                  <p className="text-gray-600">{selectedLesson.description}</p>
                </div>
              </div>
              <div className="prose max-w-none">{selectedLesson.content}</div>
              <div className="mt-8 flex justify-end">
                {!completedLessons.includes(selectedLesson.id) && (
                  <Button onClick={() => handleCompleteLesson(selectedLesson.id)}>
                    Mark as Complete (+50 XP)
                  </Button>
                )}
                {completedLessons.includes(selectedLesson.id) && (
                  <div className="flex items-center text-green-600">
                    <CheckCircle className="w-5 h-5 mr-2" />
                    <span className="font-semibold">Lesson Completed</span>
                  </div>
                )}
              </div>
            </Card>
          </motion.div>
        )}
      </div>
    </div>
  );
};
