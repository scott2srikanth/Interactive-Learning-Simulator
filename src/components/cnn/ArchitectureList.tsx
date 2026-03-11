import React from 'react';
import { motion } from 'framer-motion';
import { useCNNStore } from '../../store/cnnStore';
import { Button } from '../ui/Button';
import { Trash2, ChevronDown, ChevronUp } from 'lucide-react';
import { Layer } from '../../types/cnn';

const layerColors: Record<string, string> = {
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

export const ArchitectureList: React.FC = () => {
  const { architecture, removeLayer, selectLayer, selectedLayerId } = useCNNStore();

  if (architecture.layers.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        <p>No layers added yet</p>
        <p className="text-sm mt-2">Click the buttons above to add layers</p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <h4 className="text-sm font-semibold text-gray-700 mb-3">Architecture Layers</h4>
      {architecture.layers.map((layer, index) => (
        <LayerCard
          key={layer.id}
          layer={layer}
          index={index}
          isSelected={selectedLayerId === layer.id}
          onSelect={() => selectLayer(layer.id)}
          onRemove={() => removeLayer(layer.id)}
        />
      ))}
    </div>
  );
};

interface LayerCardProps {
  layer: Layer;
  index: number;
  isSelected: boolean;
  onSelect: () => void;
  onRemove: () => void;
}

const LayerCard: React.FC<LayerCardProps> = ({ layer, index, isSelected, onSelect, onRemove }) => {
  const [expanded, setExpanded] = React.useState(false);

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
      className={`border-2 rounded-lg p-3 cursor-pointer transition-all ${
        isSelected ? 'border-blue-500 bg-blue-50' : 'border-gray-200 bg-white hover:border-gray-300'
      }`}
      onClick={onSelect}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3 flex-1">
          <div
            className="w-10 h-10 rounded-lg flex items-center justify-center text-white text-lg font-bold"
            style={{ backgroundColor: layerColors[layer.type] }}
          >
            {layerIcons[layer.type] || index + 1}
          </div>
          <div className="flex-1">
            <div className="flex items-center space-x-2">
              <h5 className="font-semibold text-gray-900">{layer.type.toUpperCase()}</h5>
              <span className="text-xs text-gray-500">#{index + 1}</span>
            </div>
            <p className="text-xs text-gray-600 mt-0.5">{getLayerSummary(layer)}</p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <Button
            size="sm"
            variant="ghost"
            onClick={(e) => {
              e.stopPropagation();
              setExpanded(!expanded);
            }}
          >
            {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={(e) => {
              e.stopPropagation();
              onRemove();
            }}
          >
            <Trash2 className="w-4 h-4 text-red-600" />
          </Button>
        </div>
      </div>

      {expanded && (
        <motion.div
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: 'auto', opacity: 1 }}
          exit={{ height: 0, opacity: 0 }}
          className="mt-3 pt-3 border-t border-gray-200"
        >
          <div className="space-y-2 text-sm">
            {layer.type === 'conv2d' && (
              <>
                <div className="flex justify-between">
                  <span className="text-gray-600">Filters:</span>
                  <span className="font-medium">{layer.config.filters}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Kernel Size:</span>
                  <span className="font-medium">{layer.config.kernelSize}x{layer.config.kernelSize}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Stride:</span>
                  <span className="font-medium">{layer.config.stride}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Padding:</span>
                  <span className="font-medium">{layer.config.padding}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Activation:</span>
                  <span className="font-medium">{layer.config.activation}</span>
                </div>
              </>
            )}
            {(layer.type === 'maxpool' || layer.type === 'avgpool') && (
              <>
                <div className="flex justify-between">
                  <span className="text-gray-600">Pool Size:</span>
                  <span className="font-medium">{layer.config.poolSize}x{layer.config.poolSize}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Stride:</span>
                  <span className="font-medium">{layer.config.stride}</span>
                </div>
              </>
            )}
            {layer.type === 'dense' && (
              <>
                <div className="flex justify-between">
                  <span className="text-gray-600">Units:</span>
                  <span className="font-medium">{layer.config.units}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Activation:</span>
                  <span className="font-medium">{layer.config.activation}</span>
                </div>
              </>
            )}
          </div>
        </motion.div>
      )}
    </motion.div>
  );
};

function getLayerSummary(layer: Layer): string {
  switch (layer.type) {
    case 'conv2d':
      return `${layer.config.filters} filters, ${layer.config.kernelSize}x${layer.config.kernelSize}`;
    case 'maxpool':
    case 'avgpool':
      return `${layer.config.poolSize}x${layer.config.poolSize} pool`;
    case 'dense':
      return `${layer.config.units} units`;
    case 'flatten':
      return 'Converts to 1D';
    case 'softmax':
      return 'Classification output';
    default:
      return '';
  }
}
