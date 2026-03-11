# Neural Network Learn - Interactive Deep Learning Simulator

A comprehensive, interactive web platform for learning Deep Learning architectures through visual simulations and hands-on experimentation.

## Learning Topics

1. **ANN (Artificial Neural Networks)** - Fundamentals of feedforward networks
2. **CNN (Convolutional Neural Networks)** - Image processing and computer vision
3. **RNN/LSTM (Recurrent Networks)** - Sequential data and time series
4. **VAE (Variational Autoencoders)** - Generative models and latent spaces
5. **Transformers** - Attention mechanisms and modern NLP

## Features

### Core Functionality

- **Interactive CNN Builder**: Drag-and-drop interface to build CNN architectures using React Flow
- **Architecture Templates**: Pre-built architectures (Simple CNN, LeNet-5, Deep CNN, Edge Detector) for quick start
- **Real-Time Simulation Engine**: Pure TypeScript implementation of CNN operations (no GPU required)
- **Feature Map Visualization**: Visual inspection of intermediate outputs at each layer
- **RGB Color Image Support**: Full support for 3-channel color images alongside grayscale
- **Kernel Editor**: Edit and experiment with convolution kernels in real-time
- **Training Simulation**: Simulated training with live loss and accuracy graphs
- **Step-by-Step Animation**: Watch convolution and pooling operations frame-by-frame

### Learning System

- **6 Interactive Lessons**:
  1. Images as Data
  2. Convolution Intuition
  3. Filters and Kernels
  4. Activation Functions
  5. Pooling Layers
  6. CNN Architecture

- **Gamification**:
  - XP and leveling system
  - Badges and achievements
  - Challenges with rewards
  - Leaderboard

### Technical Features

- **CNN Operations**:
  - Convolution (2D)
  - Pooling (Max/Average)
  - Activation functions (ReLU, Sigmoid, Tanh)
  - Dense layers
  - Softmax
  - Flatten

- **Pre-built Kernels**:
  - Edge detection
  - Vertical/Horizontal edge detection
  - Sharpen
  - Blur

## Tech Stack

- **Framework**: React + TypeScript + Vite
- **State Management**: Zustand
- **UI Components**: TailwindCSS + Custom components
- **Visualization**:
  - React Flow (architecture builder)
  - Recharts (training graphs)
  - HTML Canvas (feature maps)
- **Animation**: Framer Motion
- **Routing**: React Router
- **Database**: Supabase (user progress, leaderboard)

## Project Structure

```
src/
├── components/
│   ├── cnn/                  # CNN-specific components
│   │   ├── CNNBuilder.tsx
│   │   ├── KernelEditor.tsx
│   │   └── TrainingDashboard.tsx
│   ├── ui/                   # Reusable UI components
│   │   ├── Button.tsx
│   │   └── Card.tsx
│   └── visualization/        # Visualization components
│       └── FeatureMapGrid.tsx
├── lib/
│   ├── cnn/                  # CNN simulation engine
│   │   ├── activations.ts
│   │   ├── convolution.ts
│   │   ├── pooling.ts
│   │   ├── dense.ts
│   │   ├── matrix.ts
│   │   └── forwardPass.ts
│   ├── datasets.ts           # Sample datasets (grayscale + RGB)
│   ├── architectureTemplates.ts  # Pre-built architectures
│   └── supabase.ts           # Supabase client
├── pages/                    # Application pages
│   ├── Landing.tsx
│   ├── Lab.tsx
│   ├── Lessons.tsx
│   ├── Dashboard.tsx
│   └── Leaderboard.tsx
├── store/                    # Zustand stores
│   ├── cnnStore.ts
│   └── userStore.ts
└── types/                    # TypeScript types
    └── cnn.ts
```

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn

### Installation

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

### Environment Variables

Create a `.env` file with your Supabase credentials:

```
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
```

## CNN Simulation Engine

The platform includes a complete CNN simulation engine written in TypeScript:

### Matrix Operations
- 2D/3D matrix creation and manipulation
- Padding, normalization, flattening
- Random weight initialization

### Convolution
- 2D convolution with configurable stride and padding
- Support for multiple filters
- Pre-built kernels for common operations

### Activation Functions
- ReLU: `f(x) = max(0, x)`
- Sigmoid: `f(x) = 1/(1+e^-x)`
- Tanh: `f(x) = tanh(x)`
- Softmax: for classification output

### Pooling
- Max pooling: takes maximum value in window
- Average pooling: takes average value in window

### Forward Pass
Orchestrates the complete forward propagation through the network, processing each layer in sequence.

## User Features

### Dashboard
- Track your learning progress
- View earned badges and achievements
- See active challenges
- Monitor XP and level

### Lab
- Load pre-built architecture templates for quick start
- Build custom CNN architectures with drag-and-drop
- Run simulations on sample images (grayscale and RGB)
- Visualize feature maps at each layer
- Experiment with kernel operations
- Train and evaluate models
- Step-by-step animations showing convolution/pooling operations

### Lessons
- Structured learning path from basics to advanced
- Interactive visualizations
- Earn XP by completing lessons

### Leaderboard
- Compare your progress with other learners
- Compete for the top spot

## Database Schema

### user_progress
Tracks individual user learning progress:
- XP and level
- Completed lessons
- Earned badges
- Completed challenges

### leaderboard
Public leaderboard data:
- Username and rank
- XP and level
- Badge count

## Contributing

This is an educational platform designed to help people learn CNNs visually. Contributions welcome!

## License

MIT License

## Acknowledgments

Inspired by TensorFlow Playground and Brilliant.org's interactive learning approach.
