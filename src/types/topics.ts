export type TopicId = 'ann' | 'cnn' | 'rnn' | 'vae' | 'transformers';

export interface Topic {
  id: TopicId;
  name: string;
  description: string;
  icon: string;
  color: string;
  gradient: string;
}

export const TOPICS: Topic[] = [
  {
    id: 'ann',
    name: 'Artificial Neural Networks',
    description: 'Learn the fundamentals of feedforward networks, backpropagation, and gradient descent',
    icon: 'Brain',
    color: 'blue',
    gradient: 'from-blue-500 to-cyan-500'
  },
  {
    id: 'cnn',
    name: 'Convolutional Neural Networks',
    description: 'Explore image processing, convolution operations, and computer vision architectures',
    icon: 'Image',
    color: 'green',
    gradient: 'from-green-500 to-emerald-500'
  },
  {
    id: 'rnn',
    name: 'RNN & LSTM',
    description: 'Master sequential data processing, time series analysis, and recurrent architectures',
    icon: 'Activity',
    color: 'orange',
    gradient: 'from-orange-500 to-amber-500'
  },
  {
    id: 'vae',
    name: 'Variational Autoencoders',
    description: 'Understand generative models, latent spaces, and probabilistic encoding',
    icon: 'Sparkles',
    color: 'pink',
    gradient: 'from-pink-500 to-rose-500'
  },
  {
    id: 'transformers',
    name: 'Transformers',
    description: 'Discover attention mechanisms, self-attention, and modern NLP architectures',
    icon: 'Zap',
    color: 'violet',
    gradient: 'from-violet-500 to-purple-500'
  }
];
