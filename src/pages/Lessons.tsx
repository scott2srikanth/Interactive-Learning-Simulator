import React, { useState } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { useUserStore } from '../store/userStore';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { CheckCircle, Circle, ArrowLeft } from 'lucide-react';
import { motion } from 'framer-motion';
import { TOPICS } from '../types/topics';

interface Lesson {
  id: string;
  title: string;
  description: string;
  icon: string;
  content: string[];
}

const ALL_LESSONS: Record<string, Lesson[]> = {
  cnn: [
    { id: 'cnn-1', title: 'Images as Data', description: 'How computers see images as matrices of numbers', icon: '🖼️', content: ['Images are grids of numbers called pixels. Grayscale images have one number per pixel (0=black, 1=white). Color images have 3 channels: Red, Green, Blue.', 'A 28×28 grayscale image = 784 numbers. A 28×28 RGB image = 2,352 numbers (28×28×3). The CNN must process all these numbers to understand the image.', 'The key insight: nearby pixels are related. A pixel next to a white pixel is likely also light. CNNs exploit this spatial structure.'] },
    { id: 'cnn-2', title: 'Convolution Operation', description: 'How sliding filters detect patterns in images', icon: '🔍', content: ['Convolution slides a small matrix (kernel/filter) across the image. At each position, it multiplies overlapping values and sums them.', 'For a 3×3 kernel on a 5×5 image with stride 1: 9 multiplications + sum = 1 output value. Slide across entire image to get output feature map.', 'Different kernels detect different features: vertical edges, horizontal edges, corners, textures. The network LEARNS these kernels during training.', 'Key parameters: kernel size (3×3, 5×5), stride (step size), padding (zeros around edges), number of filters (output channels).'] },
    { id: 'cnn-3', title: 'Filters and Feature Maps', description: 'How different kernels detect edges, textures, shapes', icon: '🎨', content: ['Edge detection kernels have positive values on one side and negative on the other. When sliding over an edge, the sum is large; on flat areas, it cancels out.', 'Blur kernels average neighboring pixels. Sharpen kernels enhance differences. Emboss kernels create 3D-like effects.', 'Each filter produces one feature map (output channel). 32 filters produce 32 feature maps, each detecting a different pattern.'] },
    { id: 'cnn-4', title: 'Activation Functions', description: 'Adding non-linearity with ReLU, Sigmoid, Tanh', icon: '⚡', content: ['Without activation functions, stacking layers is just multiplication — equivalent to a single layer. Activations add non-linearity.', 'ReLU: f(x) = max(0, x). Simple, fast, most popular. Sets negatives to zero, keeps positives unchanged.', 'Sigmoid: f(x) = 1/(1+e^-x). Squashes to [0,1]. Used for probabilities. Tanh: f(x) = tanh(x). Squashes to [-1,1]. Zero-centered.'] },
    { id: 'cnn-5', title: 'Pooling Layers', description: 'Reducing spatial dimensions while preserving features', icon: '⬇️', content: ['Pooling reduces spatial size. A 2×2 max pool with stride 2 halves each dimension (75% reduction).', 'Max pooling takes the maximum value in each window. Preserves strongest feature detections. Most commonly used.', 'Average pooling takes the mean. Smoother output. Benefits: reduces computation, provides translation invariance, prevents overfitting.'] },
    { id: 'cnn-6', title: 'CNN Architecture', description: 'Putting it all together: Conv→Pool→Dense→Softmax', icon: '🏗️', content: ['Typical CNN: Input → [Conv → ReLU → Pool] × N → Flatten → Dense → Softmax → Classification.', 'Early layers detect low-level features (edges). Middle layers combine into textures and shapes. Deep layers recognize objects.', 'Famous architectures: LeNet-5 (1998), AlexNet (2012), VGG (2014), ResNet (2015, skip connections), EfficientNet (2019).'] },
  ],
  ann: [
    { id: 'ann-1', title: 'What is a Neural Network?', description: 'Neurons, weights, and the biological inspiration', icon: '🧠', content: ['A neural network is inspired by the brain. Artificial neurons receive inputs, multiply by weights, sum, add bias, and apply activation.', 'Single neuron: output = activation(w₁x₁ + w₂x₂ + ... + wₙxₙ + bias). This is a linear combination followed by non-linearity.', 'Networks are organized in layers: Input (raw features), Hidden (learned representations), Output (predictions).'] },
    { id: 'ann-2', title: 'Forward Propagation', description: 'How data flows through the network layer by layer', icon: '➡️', content: ['Data enters the input layer and flows forward. Each neuron computes: z = Σ(wᵢ × xᵢ) + b, then a = activation(z).', 'Layer 1 outputs become Layer 2 inputs. This continues until the output layer produces the final prediction.', 'The network transforms input through increasingly abstract representations. Each layer extracts higher-level features.'] },
    { id: 'ann-3', title: 'Activation Functions', description: 'ReLU, Sigmoid, Tanh — why non-linearity matters', icon: '⚡', content: ['Without activations, a deep network collapses to a single linear transformation. Non-linearity gives neural networks their power.', 'ReLU: max(0, x). Fast, avoids vanishing gradients. Most popular for hidden layers.', 'Sigmoid: 1/(1+e^-x) for binary classification. Softmax: for multi-class, converts scores to probabilities summing to 1.'] },
    { id: 'ann-4', title: 'Loss Functions & Training', description: 'How the network measures and reduces errors', icon: '📉', content: ['Loss function measures how wrong predictions are. MSE for regression, Cross-Entropy for classification.', 'Training goal: minimize loss by adjusting weights through backpropagation and gradient descent.', 'Learning rate controls step size. Too large causes overshooting. Too small means slow convergence.'] },
    { id: 'ann-5', title: 'Backpropagation', description: 'How gradients flow backward to update weights', icon: '🔄', content: ['Backpropagation computes the gradient of loss with respect to each weight using the chain rule of calculus.', 'Starting from output, gradients flow backward. Each weight is updated: w_new = w_old - lr × gradient.', 'Chain rule: ∂L/∂w = ∂L/∂a × ∂a/∂z × ∂z/∂w. Each layer multiplies by its local gradient.'] },
    { id: 'ann-6', title: 'Decision Boundaries', description: 'How networks separate classes in feature space', icon: '🎯', content: ['A single neuron draws a linear boundary. It can only solve linearly separable problems.', 'Hidden layers create non-linear boundaries. More neurons = more complex boundaries. XOR requires at least one hidden layer.', 'The decision boundary is where output = 0.5. Training shapes this boundary to fit the data.'] },
  ],
  rnn: [
    { id: 'rnn-1', title: 'Sequential Data', description: 'Why order matters: time series, text, and sequences', icon: '📊', content: ['Some data has natural order: words in a sentence, stock prices over time. Standard neural networks ignore this order.', 'RNNs process sequences one element at a time, maintaining a memory (hidden state) that carries information from previous steps.', 'Examples: language modeling, sentiment analysis, machine translation, speech recognition, time series forecasting.'] },
    { id: 'rnn-2', title: 'RNN Cell', description: 'The recurrent connection and hidden state', icon: '🔄', content: ['At each timestep t, the RNN takes input x_t and previous hidden state h_{t-1}, producing new hidden state h_t.', 'h_t = tanh(W_xh · x_t + W_hh · h_{t-1} + b). The hidden state encodes everything seen so far.', 'The SAME weights are used at every timestep — this is weight sharing across time.'] },
    { id: 'rnn-3', title: 'Vanishing Gradients', description: 'Why simple RNNs struggle with long sequences', icon: '📉', content: ['During backpropagation through time, gradients are multiplied at each timestep. Values < 1 shrink to zero.', 'Early timesteps receive almost no gradient — the network cannot learn long-range dependencies.', 'This is the vanishing gradient problem. Simple RNNs work for short sequences but fail on long ones.'] },
    { id: 'rnn-4', title: 'LSTM Architecture', description: 'Gates that control information flow', icon: '🧬', content: ['LSTM has a cell state (long-term memory) and three gates controlling information flow.', 'Forget Gate: decides what to REMOVE. Input Gate + Candidate: decides what to ADD. Output Gate: what to OUTPUT.', 'The cell state update is additive (not multiplicative), preventing vanishing gradients.'] },
    { id: 'rnn-5', title: 'LSTM Cell State', description: 'The conveyor belt of long-term memory', icon: '🔗', content: ['Cell state runs through the entire sequence like a conveyor belt. Information flows unchanged.', 'c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t. Forget gate removes, input gate adds.', 'This additive update prevents vanishing gradients. Gradients flow through the cell state unimpeded.'] },
    { id: 'rnn-6', title: 'Applications & Variants', description: 'GRU, Bidirectional RNNs, and real-world uses', icon: '🚀', content: ['GRU: simplified LSTM with 2 gates. Often performs comparably with fewer parameters.', 'Bidirectional RNN: processes forward AND backward. Useful when both-direction context matters.', 'Applications: chatbots, speech recognition, music generation, handwriting synthesis, video captioning.'] },
  ],
  vae: [
    { id: 'vae-1', title: 'Autoencoders', description: 'Compressing and reconstructing data', icon: '🔄', content: ['An autoencoder compresses (encodes) data into a compact representation, then reconstructs (decodes) it back.', 'Input → Encoder → Latent Code (bottleneck) → Decoder → Reconstruction. The bottleneck forces learning essential features.', 'If reconstruction is good, the latent code captures the important information. This is unsupervised learning.'] },
    { id: 'vae-2', title: 'The Latent Space', description: 'Where similar data clusters together', icon: '🗺️', content: ['The latent space is the compressed representation. Each input maps to a point in this space.', 'Similar inputs map to nearby points. With a 2D latent space, you can visualize clustering.', 'Moving in latent space changes the generated output smoothly — this is what enables generation.'] },
    { id: 'vae-3', title: 'Why Variational?', description: 'From points to probability distributions', icon: '🎲', content: ['Regular autoencoders map to single points, creating gaps in latent space where decoding produces garbage.', 'VAEs map to distributions (mean μ and variance σ²). This fills latent space uniformly.', 'Probabilistic encoding creates a smooth, continuous latent space that can be sampled for generation.'] },
    { id: 'vae-4', title: 'Reparameterization Trick', description: 'Making sampling differentiable: z = μ + σ × ε', icon: '🎯', content: ['Problem: cannot backpropagate through random sampling.', 'Solution: z = μ + σ × ε, where ε ~ N(0,1). Randomness is external — gradients flow through μ and σ.', 'This trick makes VAEs trainable. Without it, we could not optimize the encoder.'] },
    { id: 'vae-5', title: 'VAE Loss Function', description: 'Reconstruction loss + KL divergence', icon: '📊', content: ['VAE loss = Reconstruction Loss + KL Divergence.', 'Reconstruction Loss (MSE): how well the decoder reproduces input. KL Divergence: how close q(z|x) is to N(0,1).', 'Together they form the ELBO. Balancing them is key: too much KL ignores input, too little gives poor generation.'] },
    { id: 'vae-6', title: 'Generation & Interpolation', description: 'Creating new data from the latent space', icon: '✨', content: ['The decoder generates new data by decoding random latent points. Sample z ~ N(0,1), decode to get new image.', 'Interpolation: encode two inputs, interpolate between latent codes, decode intermediates. Smooth transitions!', 'Applications: face generation, drug design, music composition, data augmentation, anomaly detection.'] },
  ],
  transformers: [
    { id: 'tf-1', title: 'Beyond Sequences', description: 'Why attention replaced RNNs', icon: '⚡', content: ['RNNs process tokens one-by-one, creating a bottleneck. Long-range dependencies are hard. Training is slow.', 'Transformers process ALL tokens simultaneously through self-attention. Every token directly attends to every other.', 'This enables massive parallelism and captures long-range dependencies effortlessly.'] },
    { id: 'tf-2', title: 'Token Embeddings', description: 'Converting words into vectors', icon: '📝', content: ['Words are converted to dense vectors. Similar words have similar embeddings.', 'Embedding matrix: [vocab_size × d_model]. Each row is one token\'s embedding, learned during training.', 'The embedding dimension (d_model) is key. GPT-3 uses d_model=12288.'] },
    { id: 'tf-3', title: 'Positional Encoding', description: 'Giving the model a sense of word order', icon: '🌊', content: ['Transformers have no inherent order. Positional encoding adds position information.', 'Sine/cosine encoding: each position gets a unique signature. Nearby positions have similar encodings.', 'The model learns to use relative positions from these wave patterns.'] },
    { id: 'tf-4', title: 'Self-Attention (Q, K, V)', description: 'The core mechanism of Transformers', icon: '🔑', content: ['Each token is projected into Query (Q), Key (K), Value (V) vectors via learned weight matrices.', 'Q = "what am I looking for?", K = "what do I contain?", V = "what information do I provide?"', 'Score = Q · K^T / √d_k → softmax → weights. Output = weights × V. Each token gets a context-aware representation.'] },
    { id: 'tf-5', title: 'Multi-Head Attention', description: 'Multiple attention patterns in parallel', icon: '👁️', content: ['One head learns ONE type of relationship. Multiple heads learn different relationships in parallel.', 'Each head has its own Q, K, V weights. One might attend to syntax, another semantics, another proximity.', 'Outputs concatenated and projected through W_O for a richer representation.'] },
    { id: 'tf-6', title: 'Transformer Block', description: 'Attention + FFN + Residuals + LayerNorm', icon: '🏗️', content: ['Block: Multi-Head Attention → Add & LayerNorm → Feed-Forward → Add & LayerNorm.', 'Residual connections: output = layer(x) + x. Allows direct gradient flow. Prevents degradation.', 'FFN: two linear layers with ReLU applied per position. Adds non-linear transformation capacity.'] },
  ],
};

const TOPIC_GRADIENTS: Record<string, string> = {
  cnn: 'from-green-50 via-white to-emerald-50', ann: 'from-blue-50 via-white to-cyan-50',
  rnn: 'from-orange-50 via-white to-amber-50', vae: 'from-pink-50 via-white to-rose-50',
  transformers: 'from-violet-50 via-white to-purple-50',
};

export const Lessons: React.FC = () => {
  const { topicId } = useParams();
  const navigate = useNavigate();
  const currentTopic = TOPICS.find(t => t.id === topicId);
  const { completedLessons, completeLesson } = useUserStore();
  const [selectedLesson, setSelectedLesson] = useState<Lesson | null>(null);
  const lessons = ALL_LESSONS[topicId || ''] || [];
  const gradient = TOPIC_GRADIENTS[topicId || ''] || 'from-gray-50 via-white to-gray-50';

  if (!lessons.length) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center">
        <Card className="max-w-md"><div className="p-8 text-center"><h2 className="text-2xl font-bold text-white mb-4">Coming Soon</h2><Button onClick={() => navigate('/topics')}>Back to Topics</Button></div></Card>
      </div>
    );
  }

  return (
    <div className={`min-h-screen bg-gradient-to-br ${gradient}`}>
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-4">
              <Button variant="ghost" onClick={() => navigate('/topics')}><ArrowLeft className="w-4 h-4 mr-2" />Topics</Button>
              <h1 className="text-2xl font-bold text-gray-900">{currentTopic?.name} Lessons</h1>
            </div>
            <div className="flex space-x-2">
              <Link to="/dashboard"><Button variant="ghost">Dashboard</Button></Link>
              <Link to={`/topics/${topicId}/lab`}><Button variant="primary">Go to Lab</Button></Link>
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
                <motion.div key={lesson.id} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: idx * 0.1 }}>
                  <Card className="h-full hover:shadow-xl transition-shadow cursor-pointer">
                    <div onClick={() => setSelectedLesson(lesson)}>
                      <div className="flex items-start justify-between mb-4">
                        <div className="text-3xl">{lesson.icon}</div>
                        {isCompleted ? <CheckCircle className="w-6 h-6 text-green-600" /> : <Circle className="w-6 h-6 text-gray-300" />}
                      </div>
                      <h3 className="text-xl font-semibold text-gray-900 mb-2">{lesson.title}</h3>
                      <p className="text-gray-600 text-sm mb-4">{lesson.description}</p>
                      <Button variant="outline" size="sm" className="w-full">{isCompleted ? 'Review Lesson' : 'Start Lesson'}</Button>
                    </div>
                  </Card>
                </motion.div>
              );
            })}
          </div>
        ) : (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
            <Card>
              <div className="mb-6"><Button variant="ghost" onClick={() => setSelectedLesson(null)}>← Back to Lessons</Button></div>
              <div className="flex items-center space-x-4 mb-6">
                <div className="text-4xl">{selectedLesson.icon}</div>
                <div><h2 className="text-3xl font-bold text-gray-900">{selectedLesson.title}</h2><p className="text-gray-600">{selectedLesson.description}</p></div>
              </div>
              <div className="prose max-w-none space-y-4">
                {selectedLesson.content.map((para, i) => (<p key={i} className="text-gray-700 leading-relaxed">{para}</p>))}
              </div>
              <div className="mt-8 pt-6 border-t border-gray-200 flex items-center justify-between">
                <Link to={`/topics/${topicId}/lab`}><Button variant="outline">Try in Lab →</Button></Link>
                {!completedLessons.includes(selectedLesson.id) ? (
                  <Button onClick={() => completeLesson(selectedLesson.id)}>Mark as Complete (+50 XP)</Button>
                ) : (
                  <div className="flex items-center text-green-600"><CheckCircle className="w-5 h-5 mr-2" /><span className="font-semibold">Completed</span></div>
                )}
              </div>
            </Card>
          </motion.div>
        )}
      </div>
    </div>
  );
};
