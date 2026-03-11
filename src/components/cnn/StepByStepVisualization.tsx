import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';
import { FeatureMapGrid } from '../visualization/FeatureMapGrid';
import { ConvolutionAnimation } from './ConvolutionAnimation';
import { PoolingAnimation } from './PoolingAnimation';
import { useCNNStore } from '../../store/cnnStore';
import { Matrix2D, Matrix3D } from '../../types/cnn';
import { Play, Pause, SkipForward, SkipBack, RotateCcw, Zap } from 'lucide-react';
import { convolve2D } from '../../lib/cnn/convolution';
import { maxPool2D, avgPool2D } from '../../lib/cnn/pooling';

// Seeded random number generator for consistent kernels
function seededRandom(seed: number) {
  let x = Math.sin(seed++) * 10000;
  return x - Math.floor(x);
}

// Generate consistent kernel based on layer ID
function generateKernelForLayer(layerId: string, kernelSize: number): Matrix2D {
  const seed = layerId.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);

  // Use predefined kernels for common sizes
  if (kernelSize === 3) {
    const kernels = [
      // Edge detection
      [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
      // Sharpen
      [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
      // Vertical edge
      [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
      // Horizontal edge
      [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
      // Blur
      [[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]],
    ];
    return kernels[seed % kernels.length];
  } else if (kernelSize === 5) {
    return [
      [-1, -1, -1, -1, -1],
      [-1, 2, 2, 2, -1],
      [-1, 2, 8, 2, -1],
      [-1, 2, 2, 2, -1],
      [-1, -1, -1, -1, -1],
    ];
  }

  // For other sizes, generate seeded random kernel
  const kernel: Matrix2D = [];
  for (let i = 0; i < kernelSize; i++) {
    const row: number[] = [];
    for (let j = 0; j < kernelSize; j++) {
      row.push(seededRandom(seed + i * kernelSize + j) * 2 - 1);
    }
    kernel.push(row);
  }
  return kernel;
}

export const StepByStepVisualization: React.FC = () => {
  const { architecture, layerOutputs } = useCNNStore();
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showDetails, setShowDetails] = useState(false);

  const steps = Array.from(layerOutputs.entries()).filter(([_, output]) => {
    return Array.isArray(output[0]) && Array.isArray(output[0][0]);
  });

  // Memoize kernels for each conv layer to ensure consistency
  const layerKernels = useMemo(() => {
    const kernels = new Map<string, Matrix2D>();
    architecture.layers.forEach(layer => {
      if (layer.type === 'conv2d') {
        kernels.set(layer.id, generateKernelForLayer(layer.id, layer.config.kernelSize));
      }
    });
    return kernels;
  }, [architecture.layers]);

  useEffect(() => {
    if (isPlaying && currentStep < steps.length - 1) {
      const timer = setTimeout(() => {
        setCurrentStep((prev) => prev + 1);
      }, 2500);
      return () => clearTimeout(timer);
    } else if (currentStep >= steps.length - 1) {
      setIsPlaying(false);
    }
  }, [isPlaying, currentStep, steps.length]);

  const handlePlay = () => {
    if (currentStep >= steps.length - 1) {
      setCurrentStep(0);
    }
    setIsPlaying(true);
  };

  const handlePause = () => {
    setIsPlaying(false);
  };

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep((prev) => prev + 1);
    }
  };

  const handlePrev = () => {
    if (currentStep > 0) {
      setCurrentStep((prev) => prev - 1);
    }
  };

  const handleReset = () => {
    setCurrentStep(0);
    setIsPlaying(false);
  };

  if (steps.length === 0) {
    return (
      <Card>
        <div className="text-center py-16">
          <Zap className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <p className="text-lg text-gray-600 mb-2">No simulation data available</p>
          <p className="text-sm text-gray-500 mb-4">
            {architecture.layers.length === 0
              ? 'Add layers in the Architecture Builder tab first'
              : !architecture.input
              ? 'Select an input image from the left sidebar'
              : 'Click the Run button in the Architecture Builder tab to generate visualization data'
            }
          </p>
          {layerOutputs.size > 0 && (
            <details className="mt-4 text-xs text-left bg-gray-100 p-3 rounded">
              <summary className="cursor-pointer font-semibold">Debug Info</summary>
              <div className="mt-2 space-y-1">
                <p>Architecture layers: {architecture.layers.length}</p>
                <p>Layer outputs count: {layerOutputs.size}</p>
                <p>Has input: {architecture.input ? 'Yes' : 'No'}</p>
                <p>Output layer IDs: {Array.from(layerOutputs.keys()).join(', ')}</p>
              </div>
            </details>
          )}
        </div>
      </Card>
    );
  }

  const [layerId, output] = steps[currentStep];
  const layer = architecture.layers.find((l) => l.id === layerId);
  const layerIndex = architecture.layers.findIndex((l) => l.id === layerId);

  const prevOutput = currentStep > 0 ? steps[currentStep - 1][1] : (architecture.input || null);

  return (
    <Card>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-2xl font-bold text-gray-900">Step-by-Step Visualization</h3>
            <p className="text-sm text-gray-600 mt-1">
              Step {currentStep + 1} of {steps.length} • {layer?.type.toUpperCase()} Layer
            </p>
          </div>

          <div className="flex items-center gap-2">
            <Button size="sm" variant="outline" onClick={handleReset} disabled={currentStep === 0}>
              <RotateCcw className="w-4 h-4" />
            </Button>
            <Button size="sm" variant="outline" onClick={handlePrev} disabled={currentStep === 0}>
              <SkipBack className="w-4 h-4" />
            </Button>
            {isPlaying ? (
              <Button size="sm" onClick={handlePause}>
                <Pause className="w-4 h-4 mr-1" />
                Pause
              </Button>
            ) : (
              <Button size="sm" onClick={handlePlay}>
                <Play className="w-4 h-4 mr-1" />
                Play
              </Button>
            )}
            <Button
              size="sm"
              variant="outline"
              onClick={handleNext}
              disabled={currentStep === steps.length - 1}
            >
              <SkipForward className="w-4 h-4" />
            </Button>
          </div>
        </div>

        <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
          <motion.div
            className="bg-gradient-to-r from-blue-500 to-green-500 h-3"
            initial={{ width: 0 }}
            animate={{ width: `${((currentStep + 1) / steps.length) * 100}%` }}
            transition={{ duration: 0.5, ease: 'easeInOut' }}
          />
        </div>

        <div className="grid grid-cols-5 gap-2 mb-4">
          {steps.map((_, idx) => (
            <motion.button
              key={idx}
              onClick={() => setCurrentStep(idx)}
              className={`h-2 rounded-full transition-all ${
                idx === currentStep
                  ? 'bg-blue-600'
                  : idx < currentStep
                  ? 'bg-green-500'
                  : 'bg-gray-300'
              }`}
              whileHover={{ scale: 1.2 }}
              whileTap={{ scale: 0.9 }}
            />
          ))}
        </div>

        <AnimatePresence mode="wait">
          <motion.div
            key={currentStep}
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -50 }}
            transition={{ duration: 0.4 }}
            className="space-y-6"
          >
            <div className="bg-gradient-to-br from-blue-50 via-white to-green-50 p-6 rounded-xl border-2 border-blue-200 shadow-lg">
              <div className="flex items-start justify-between mb-4">
                <div className="flex-1">
                  <div className="flex items-center space-x-3 mb-2">
                    <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center text-white font-bold text-lg">
                      {layerIndex + 1}
                    </div>
                    <div>
                      <h4 className="text-2xl font-bold text-gray-900">{layer?.type.toUpperCase()}</h4>
                      <p className="text-sm text-gray-600">Processing Layer {layerIndex + 1}</p>
                    </div>
                  </div>
                  <p className="text-gray-700 mb-4">{getLayerDescription(layer?.type || '')}</p>

                  {layer && layer.type === 'conv2d' && (
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-2 mt-3">
                      <InfoBadge label="Filters" value={layer.config.filters} color="blue" />
                      <InfoBadge
                        label="Kernel"
                        value={`${layer.config.kernelSize}×${layer.config.kernelSize}`}
                        color="green"
                      />
                      <InfoBadge label="Stride" value={layer.config.stride} color="yellow" />
                      <InfoBadge label="Padding" value={layer.config.padding} color="purple" />
                      <InfoBadge label="Activation" value={layer.config.activation} color="pink" />
                    </div>
                  )}
                  {layer && (layer.type === 'maxpool' || layer.type === 'avgpool') && (
                    <div className="grid grid-cols-3 gap-2 mt-3">
                      <InfoBadge
                        label="Pool Size"
                        value={`${layer.config.poolSize}×${layer.config.poolSize}`}
                        color="orange"
                      />
                      <InfoBadge label="Stride" value={layer.config.stride} color="yellow" />
                      <InfoBadge label="Type" value={layer.config.type} color="red" />
                    </div>
                  )}
                  {layer && layer.type === 'dense' && (
                    <div className="grid grid-cols-2 gap-2 mt-3">
                      <InfoBadge label="Units" value={layer.config.units} color="pink" />
                      <InfoBadge label="Activation" value={layer.config.activation} color="purple" />
                    </div>
                  )}
                </div>
              </div>

              <div className="flex items-center space-x-2 mb-3">
                <motion.div
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ repeat: Infinity, duration: 2 }}
                  className="w-2 h-2 bg-green-500 rounded-full"
                />
                <span className="text-sm font-semibold text-gray-700">
                  {isPlaying ? 'Processing...' : 'Ready'}
                </span>
              </div>
            </div>

            {layer && layer.type === 'conv2d' && prevOutput && layerKernels.get(layerId) && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
              >
                <ConvolutionAnimation
                  input={(prevOutput as Matrix3D)[0]}
                  kernel={layerKernels.get(layerId)!}
                  output={convolve2D(
                    (prevOutput as Matrix3D)[0],
                    layerKernels.get(layerId)!,
                    layer.config.stride
                  )}
                  stride={layer.config.stride}
                />
              </motion.div>
            )}

            {layer && (layer.type === 'maxpool' || layer.type === 'avgpool') && prevOutput && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
              >
                <PoolingAnimation
                  input={(prevOutput as Matrix3D)[0]}
                  output={
                    layer.config.type === 'max'
                      ? maxPool2D((prevOutput as Matrix3D)[0], layer.config.poolSize, layer.config.stride)
                      : avgPool2D((prevOutput as Matrix3D)[0], layer.config.poolSize, layer.config.stride)
                  }
                  poolSize={layer.config.poolSize}
                  poolType={layer.config.type}
                  stride={layer.config.stride}
                />
              </motion.div>
            )}

            {layer &&
              layer.type !== 'conv2d' &&
              layer.type !== 'maxpool' &&
              layer.type !== 'avgpool' && (
                <>
                  {prevOutput && (
                    <div>
                      <h5 className="text-sm font-semibold text-gray-700 mb-3 flex items-center">
                        <span className="w-2 h-2 bg-gray-400 rounded-full mr-2"></span>
                        Input to this layer
                      </h5>
                      <div className="bg-white p-4 rounded-lg border border-gray-200">
                        <FeatureMapGrid data={prevOutput as Matrix3D} maxMaps={6} />
                      </div>
                    </div>
                  )}

                  <motion.div
                    initial={{ scale: 0.95, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ delay: 0.2, duration: 0.3 }}
                  >
                    <div className="flex items-center space-x-2 mb-3">
                      <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ repeat: isPlaying ? Infinity : 0, duration: 2, ease: 'linear' }}
                      >
                        <Zap className="w-5 h-5 text-yellow-500" />
                      </motion.div>
                      <h5 className="text-sm font-semibold text-gray-700">
                        Transformation Result (Feature Maps)
                      </h5>
                    </div>
                    <div className="bg-gradient-to-br from-yellow-50 to-orange-50 p-6 rounded-lg border-2 border-yellow-200">
                      <FeatureMapGrid data={output as Matrix3D} maxMaps={8} />
                      <div className="mt-4 text-center text-sm text-gray-600">
                        Output dimensions: {(output as Matrix3D).length} channels ×{' '}
                        {(output as Matrix3D)[0]?.length || 0} × {(output as Matrix3D)[0]?.[0]?.length || 0}
                      </div>
                    </div>
                  </motion.div>
                </>
              )}

            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowDetails(!showDetails)}
              className="w-full"
            >
              {showDetails ? 'Hide' : 'Show'} Technical Details
            </Button>

            {showDetails && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="bg-gray-900 text-gray-100 p-6 rounded-lg font-mono text-xs overflow-auto"
              >
                <div className="space-y-2">
                  <p>
                    <span className="text-green-400">Layer ID:</span> {layerId}
                  </p>
                  <p>
                    <span className="text-green-400">Layer Type:</span> {layer?.type}
                  </p>
                  <p>
                    <span className="text-green-400">Input Shape:</span> [
                    {prevOutput
                      ? `${(prevOutput as Matrix3D).length}, ${(prevOutput as Matrix3D)[0]?.length}, ${
                          (prevOutput as Matrix3D)[0]?.[0]?.length
                        }`
                      : 'N/A'}
                    ]
                  </p>
                  <p>
                    <span className="text-green-400">Output Shape:</span> [
                    {`${(output as Matrix3D).length}, ${(output as Matrix3D)[0]?.length}, ${
                      (output as Matrix3D)[0]?.[0]?.length
                    }`}
                    ]
                  </p>
                  {layer?.type === 'conv2d' && (
                    <>
                      <p>
                        <span className="text-green-400">Parameters:</span>{' '}
                        {layer.config.filters *
                          layer.config.kernelSize *
                          layer.config.kernelSize *
                          ((prevOutput as Matrix3D)?.length || 1)}
                      </p>
                      <p>
                        <span className="text-green-400">Operation:</span> 2D Convolution with {layer.config.activation}{' '}
                        activation
                      </p>
                    </>
                  )}
                </div>
              </motion.div>
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </Card>
  );
};

const InfoBadge: React.FC<{ label: string; value: string | number; color: string }> = ({
  label,
  value,
  color,
}) => {
  const colorClasses: Record<string, string> = {
    blue: 'bg-blue-100 text-blue-800 border-blue-300',
    green: 'bg-green-100 text-green-800 border-green-300',
    yellow: 'bg-yellow-100 text-yellow-800 border-yellow-300',
    purple: 'bg-purple-100 text-purple-800 border-purple-300',
    pink: 'bg-pink-100 text-pink-800 border-pink-300',
    orange: 'bg-orange-100 text-orange-800 border-orange-300',
    red: 'bg-red-100 text-red-800 border-red-300',
  };

  return (
    <div className={`px-3 py-2 rounded-lg border ${colorClasses[color]} text-center`}>
      <p className="text-xs font-medium opacity-75">{label}</p>
      <p className="text-sm font-bold">{value}</p>
    </div>
  );
};

function getLayerDescription(type: string): string {
  const descriptions: Record<string, string> = {
    input: '📥 Input layer receives the raw image data as a numerical matrix of pixel values',
    conv2d:
      '🔍 Convolution applies learnable filters that slide across the image to detect features like edges, textures, and patterns. Each filter performs element-wise multiplication with image regions.',
    maxpool:
      '⬇️ Max pooling downsamples the feature maps by taking the maximum value in each region. This reduces spatial dimensions while preserving the strongest activations.',
    avgpool:
      '📊 Average pooling downsamples by computing the average value in each region, providing smoother dimensionality reduction than max pooling.',
    flatten:
      '➡️ Flatten reshapes multi-dimensional feature maps into a single 1D vector, preparing the data for fully connected layers.',
    dense:
      '🧠 Dense (fully connected) layer where every neuron connects to all neurons in the previous layer, learning complex high-level patterns.',
    softmax:
      '📈 Softmax converts raw output scores into probabilities that sum to 1.0, enabling multi-class classification.',
  };
  return descriptions[type] || 'Processing the data through this layer';
}
