import React, { useState, useMemo } from 'react';
import { Layer, LayerType, Matrix2D, Matrix3D } from '../../types/cnn';
import { useCNNStore } from '../../store/cnnStore';
import { Button } from '../ui/Button';
import { Plus, Trash2, ChevronDown, ChevronUp, Play } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { ConvolutionAnimation } from './ConvolutionAnimation';
import { PoolingAnimation } from './PoolingAnimation';
import { FeatureMapGrid } from '../visualization/FeatureMapGrid';
import { convolve2D } from '../../lib/cnn/convolution';
import { maxPool2D, avgPool2D } from '../../lib/cnn/pooling';

const layerColors: Record<LayerType, string> = {
  input: '#10b981',
  conv2d: '#3b82f6',
  maxpool: '#f59e0b',
  avgpool: '#f59e0b',
  flatten: '#8b5cf6',
  dense: '#ec4899',
  dropout: '#6b7280',
  softmax: '#ef4444',
};

const layerIcons: Record<string, string> = {
  input: '📥',
  conv2d: '🔍',
  maxpool: '⬇️',
  avgpool: '📊',
  flatten: '➡️',
  dense: '🧠',
  dropout: '❌',
  softmax: '📈',
};

// Seeded random number generator for consistent kernels
function seededRandom(seed: number) {
  let x = Math.sin(seed++) * 10000;
  return x - Math.floor(x);
}

// Generate consistent kernel based on layer ID
function generateKernelForLayer(layerId: string, kernelSize: number): Matrix2D {
  const seed = layerId.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);

  if (kernelSize === 3) {
    const kernels = [
      [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
      [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
      [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
      [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
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

export const CNNBuilder: React.FC = () => {
  const { architecture, addLayer, removeLayer, selectLayer, selectedLayerId, runForwardPass, layerOutputs } = useCNNStore();
  const [expandedLayers, setExpandedLayers] = useState<Set<string>>(new Set());
  const [isRunning, setIsRunning] = useState(false);

  const layerKernels = useMemo(() => {
    const kernels = new Map<string, Matrix2D>();
    architecture.layers.forEach(layer => {
      if (layer.type === 'conv2d') {
        kernels.set(layer.id, generateKernelForLayer(layer.id, layer.config.kernelSize));
      }
    });
    return kernels;
  }, [architecture.layers]);

  const toggleExpanded = (layerId: string) => {
    const newExpanded = new Set(expandedLayers);
    if (newExpanded.has(layerId)) {
      newExpanded.delete(layerId);
    } else {
      newExpanded.add(layerId);
    }
    setExpandedLayers(newExpanded);
  };

  const addLayerToArchitecture = (type: LayerType) => {
    const id = `${type}-${Date.now()}`;
    const layer: Partial<Layer> = {
      id,
      type,
      name: type.toUpperCase(),
    };

    if (type === 'conv2d') {
      addLayer({
        ...layer,
        type: 'conv2d',
        config: {
          filters: 32,
          kernelSize: 3,
          stride: 1,
          padding: 'same',
          activation: 'relu',
        },
      } as Layer);
    } else if (type === 'maxpool' || type === 'avgpool') {
      addLayer({
        ...layer,
        type,
        config: {
          poolSize: 2,
          stride: 2,
          type: type === 'maxpool' ? 'max' : 'average',
        },
      } as Layer);
    } else if (type === 'dense') {
      addLayer({
        ...layer,
        type: 'dense',
        config: {
          units: 128,
          activation: 'relu',
        },
      } as Layer);
    } else {
      addLayer(layer as Layer);
    }
  };

  const clearAll = () => {
    useCNNStore.getState().clearArchitecture();
  };

  const handleRun = async () => {
    setIsRunning(true);
    try {
      await runForwardPass();
    } finally {
      setTimeout(() => setIsRunning(false), 500);
    }
  };

  return (
    <div className="h-full flex flex-col overflow-auto">
      <div className="bg-white border-b border-gray-200 p-4 flex flex-wrap items-center gap-3 sticky top-0 z-10">
        <h3 className="text-lg font-semibold text-gray-800 mr-4">Add Layers:</h3>

        <Button size="sm" onClick={() => addLayerToArchitecture('conv2d')}>
          <Plus className="w-4 h-4 mr-1" />
          Conv2D
        </Button>
        <Button size="sm" onClick={() => addLayerToArchitecture('maxpool')}>
          <Plus className="w-4 h-4 mr-1" />
          MaxPool
        </Button>
        <Button size="sm" onClick={() => addLayerToArchitecture('avgpool')}>
          <Plus className="w-4 h-4 mr-1" />
          AvgPool
        </Button>
        <Button size="sm" onClick={() => addLayerToArchitecture('flatten')}>
          <Plus className="w-4 h-4 mr-1" />
          Flatten
        </Button>
        <Button size="sm" onClick={() => addLayerToArchitecture('dense')}>
          <Plus className="w-4 h-4 mr-1" />
          Dense
        </Button>
        <Button size="sm" onClick={() => addLayerToArchitecture('softmax')}>
          <Plus className="w-4 h-4 mr-1" />
          Softmax
        </Button>

        <div className="ml-auto flex gap-2">
          <Button variant="outline" size="sm" onClick={clearAll}>
            <Trash2 className="w-4 h-4 mr-1" />
            Clear
          </Button>
          <Button
            size="sm"
            onClick={handleRun}
            disabled={architecture.layers.length === 0 || isRunning}
          >
            <Play className="w-4 h-4 mr-1" />
            {isRunning ? 'Running...' : 'Run'}
          </Button>
        </div>
      </div>

      <div className="flex-1 p-6 bg-gray-50">
        {architecture.layers.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center text-gray-500">
              <p className="text-lg mb-2">No layers added yet</p>
              <p className="text-sm">Click the buttons above to build your CNN architecture</p>
            </div>
          </div>
        ) : (
          <div className="max-w-2xl mx-auto space-y-3">
            <AnimatePresence>
              {architecture.layers.map((layer, index) => (
                <React.Fragment key={layer.id}>
                  <motion.div
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, x: -100 }}
                    transition={{ duration: 0.3, delay: index * 0.05 }}
                  >
                    <LayerCard
                      layer={layer}
                      index={index}
                      isSelected={selectedLayerId === layer.id}
                      isExpanded={expandedLayers.has(layer.id)}
                      onSelect={() => selectLayer(layer.id)}
                      onRemove={() => removeLayer(layer.id)}
                      onToggleExpand={() => toggleExpanded(layer.id)}
                      layerOutput={layerOutputs.get(layer.id)}
                      prevOutput={index > 0 ? layerOutputs.get(architecture.layers[index - 1].id) : architecture.input}
                      kernel={layerKernels.get(layer.id)}
                    />
                  </motion.div>
                  {index < architecture.layers.length - 1 && (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="flex justify-center"
                    >
                      <ChevronDown className="w-6 h-6 text-gray-400" />
                    </motion.div>
                  )}
                </React.Fragment>
              ))}
            </AnimatePresence>
          </div>
        )}
      </div>
    </div>
  );
};

interface LayerCardProps {
  layer: Layer;
  index: number;
  isSelected: boolean;
  isExpanded: boolean;
  onSelect: () => void;
  onRemove: () => void;
  onToggleExpand: () => void;
  layerOutput?: Matrix3D | number[];
  prevOutput?: Matrix3D | number[] | null;
  kernel?: Matrix2D;
}

const LayerCard: React.FC<LayerCardProps> = ({
  layer,
  index,
  isSelected,
  isExpanded,
  onSelect,
  onRemove,
  onToggleExpand,
  layerOutput,
  prevOutput,
  kernel,
}) => {
  const is3DInput = prevOutput && Array.isArray(prevOutput) && Array.isArray(prevOutput[0]) && Array.isArray(prevOutput[0][0]);
  const is3DOutput = layerOutput && Array.isArray(layerOutput) && Array.isArray(layerOutput[0]) && Array.isArray(layerOutput[0][0]);
  const hasVisualization = layerOutput && prevOutput;
  return (
    <div
      className={`border-2 rounded-lg overflow-hidden transition-all cursor-pointer ${
        isSelected ? 'border-blue-500 shadow-lg' : 'border-gray-200 hover:border-gray-300'
      }`}
      onClick={onSelect}
    >
      <div className="bg-white p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3 flex-1">
            <div
              className="w-12 h-12 rounded-lg flex items-center justify-center text-white text-2xl font-bold shadow-md"
              style={{ backgroundColor: layerColors[layer.type] }}
            >
              {layerIcons[layer.type] || index + 1}
            </div>
            <div className="flex-1">
              <div className="flex items-center space-x-2">
                <h5 className="font-bold text-lg text-gray-900">{layer.type.toUpperCase()}</h5>
                <span className="text-xs bg-gray-200 text-gray-700 px-2 py-1 rounded">
                  #{index + 1}
                </span>
              </div>
              <p className="text-sm text-gray-600 mt-1">{getLayerSummary(layer)}</p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <Button
              size="sm"
              variant="ghost"
              onClick={(e) => {
                e.stopPropagation();
                onToggleExpand();
              }}
            >
              {isExpanded ? (
                <ChevronUp className="w-5 h-5" />
              ) : (
                <ChevronDown className="w-5 h-5" />
              )}
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={(e) => {
                e.stopPropagation();
                onRemove();
              }}
            >
              <Trash2 className="w-5 h-5 text-red-600" />
            </Button>
          </div>
        </div>
      </div>

      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="bg-gray-50 border-t border-gray-200"
          >
            <div className="p-4 space-y-3">
              <h6 className="font-semibold text-sm text-gray-700 mb-2">Layer Details</h6>
              {layer.type === 'conv2d' && (
                <div className="grid grid-cols-2 gap-3">
                  <DetailItem label="Filters" value={layer.config.filters} />
                  <DetailItem
                    label="Kernel Size"
                    value={`${layer.config.kernelSize}×${layer.config.kernelSize}`}
                  />
                  <DetailItem label="Stride" value={layer.config.stride} />
                  <DetailItem label="Padding" value={layer.config.padding} />
                  <DetailItem label="Activation" value={layer.config.activation} />
                </div>
              )}
              {(layer.type === 'maxpool' || layer.type === 'avgpool') && (
                <div className="grid grid-cols-2 gap-3">
                  <DetailItem
                    label="Pool Size"
                    value={`${layer.config.poolSize}×${layer.config.poolSize}`}
                  />
                  <DetailItem label="Stride" value={layer.config.stride} />
                  <DetailItem label="Type" value={layer.config.type} />
                </div>
              )}
              {layer.type === 'dense' && (
                <div className="grid grid-cols-2 gap-3">
                  <DetailItem label="Units" value={layer.config.units} />
                  <DetailItem label="Activation" value={layer.config.activation} />
                </div>
              )}
              {layer.type === 'flatten' && (
                <p className="text-sm text-gray-600">Converts multi-dimensional data to 1D vector</p>
              )}
              {layer.type === 'softmax' && (
                <p className="text-sm text-gray-600">Converts outputs to probabilities</p>
              )}
              <div className="mt-3 p-3 bg-blue-50 rounded-lg">
                <p className="text-xs text-gray-700">{getLayerDescription(layer.type)}</p>
              </div>

              {hasVisualization && (
                <div className="mt-4 pt-4 border-t border-gray-300">
                  <h6 className="font-semibold text-sm text-gray-700 mb-3">Live Visualization</h6>

                  {layer.type === 'conv2d' && kernel && is3DInput && (
                    <div className="bg-white p-4 rounded-lg border border-gray-300">
                      <ConvolutionAnimation
                        input={(prevOutput as Matrix3D)[0]}
                        kernel={kernel}
                        output={convolve2D(
                          (prevOutput as Matrix3D)[0],
                          kernel,
                          layer.config.stride
                        )}
                        stride={layer.config.stride}
                      />
                    </div>
                  )}

                  {(layer.type === 'maxpool' || layer.type === 'avgpool') && is3DInput && (
                    <div className="bg-white p-4 rounded-lg border border-gray-300">
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
                    </div>
                  )}

                  {layer.type === 'flatten' && is3DInput && (
                    <div className="bg-white p-4 rounded-lg border border-gray-300">
                      <div className="space-y-3">
                        <div>
                          <p className="text-xs font-medium text-gray-600 mb-2">Input (3D)</p>
                          <FeatureMapGrid data={prevOutput as Matrix3D} maxMaps={6} />
                        </div>
                        <div className="flex justify-center my-2">
                          <div className="text-2xl text-gray-400">↓</div>
                        </div>
                        <div>
                          <p className="text-xs font-medium text-gray-600 mb-2">Output (1D Vector)</p>
                          <div className="bg-gray-100 p-3 rounded">
                            <p className="text-xs text-gray-700">
                              Flattened to {(layerOutput as number[]).length} values
                            </p>
                            <div className="mt-2 flex flex-wrap gap-1 max-h-20 overflow-hidden">
                              {(layerOutput as number[]).slice(0, 50).map((val, i) => (
                                <span key={i} className="text-xs bg-blue-100 px-1 rounded">
                                  {val.toFixed(2)}
                                </span>
                              ))}
                              {(layerOutput as number[]).length > 50 && (
                                <span className="text-xs text-gray-500">... and {(layerOutput as number[]).length - 50} more</span>
                              )}
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {(layer.type === 'dense' || layer.type === 'softmax') && (
                    <div className="bg-white p-4 rounded-lg border border-gray-300">
                      <div className="space-y-3">
                        <div>
                          <p className="text-xs font-medium text-gray-600 mb-2">
                            {layer.type === 'dense' ? 'Dense Layer Output' : 'Class Probabilities'}
                          </p>
                          <div className="bg-gray-100 p-3 rounded">
                            <p className="text-xs text-gray-700 mb-2">
                              {(layerOutput as number[]).length} output values
                            </p>
                            <div className="space-y-1">
                              {(layerOutput as number[]).map((val, i) => (
                                <div key={i} className="flex items-center gap-2">
                                  <span className="text-xs text-gray-600 w-12">
                                    {layer.type === 'softmax' ? `Class ${i}` : `Unit ${i}`}
                                  </span>
                                  <div className="flex-1 bg-gray-200 rounded-full h-4 overflow-hidden">
                                    <div
                                      className="bg-blue-500 h-full transition-all"
                                      style={{ width: `${layer.type === 'softmax' ? val * 100 : Math.min(Math.abs(val) * 100, 100)}%` }}
                                    />
                                  </div>
                                  <span className="text-xs font-mono text-gray-700 w-16 text-right">
                                    {val.toFixed(4)}
                                  </span>
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {layer.type !== 'conv2d' &&
                   layer.type !== 'maxpool' &&
                   layer.type !== 'avgpool' &&
                   layer.type !== 'flatten' &&
                   layer.type !== 'dense' &&
                   layer.type !== 'softmax' &&
                   is3DOutput && (
                    <div className="bg-white p-4 rounded-lg border border-gray-300">
                      <div className="space-y-3">
                        <div>
                          <p className="text-xs font-medium text-gray-600 mb-2">Output Feature Maps</p>
                          <FeatureMapGrid data={layerOutput as Matrix3D} maxMaps={6} />
                        </div>
                        <div className="text-center text-xs text-gray-600">
                          Output dimensions: {`${(layerOutput as Matrix3D).length} channels × ${(layerOutput as Matrix3D)[0]?.length || 0} × ${(layerOutput as Matrix3D)[0]?.[0]?.length || 0}`}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

const DetailItem: React.FC<{ label: string; value: string | number }> = ({ label, value }) => (
  <div className="bg-white p-2 rounded border border-gray-200">
    <p className="text-xs text-gray-600">{label}</p>
    <p className="text-sm font-semibold text-gray-900">{value}</p>
  </div>
);

function getLayerSummary(layer: Layer): string {
  switch (layer.type) {
    case 'conv2d':
      return `${layer.config.filters} filters, ${layer.config.kernelSize}×${layer.config.kernelSize} kernel`;
    case 'maxpool':
    case 'avgpool':
      return `${layer.config.poolSize}×${layer.config.poolSize} ${layer.config.type} pooling`;
    case 'dense':
      return `${layer.config.units} units, ${layer.config.activation} activation`;
    case 'flatten':
      return 'Converts to 1D vector';
    case 'softmax':
      return 'Classification probabilities';
    default:
      return '';
  }
}

function getLayerDescription(type: string): string {
  const descriptions: Record<string, string> = {
    input: 'Input layer receives the raw image data as a numerical matrix',
    conv2d:
      'Convolution applies learnable filters to detect features like edges, textures, and patterns. Each filter slides across the image performing element-wise multiplication.',
    maxpool:
      'Max pooling downsamples by taking the maximum value in each region, preserving the strongest activations and reducing spatial dimensions.',
    avgpool:
      'Average pooling downsamples by computing the average value in each region, providing a smoother dimensionality reduction.',
    flatten:
      'Flatten reshapes the multi-dimensional feature maps into a single vector, preparing data for fully connected layers.',
    dense:
      'Fully connected layer where every neuron connects to all neurons in the previous layer, learning complex patterns and relationships.',
    softmax:
      'Softmax converts raw outputs into probabilities that sum to 1, enabling multi-class classification.',
  };
  return descriptions[type] || 'Processes the data';
}
