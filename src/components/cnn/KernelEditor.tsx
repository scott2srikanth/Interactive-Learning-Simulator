import React, { useState } from 'react';
import { Matrix2D } from '../../types/cnn';
import {
  createEdgeDetectionKernel,
  createVerticalEdgeKernel,
  createHorizontalEdgeKernel,
  createSharpenKernel,
  createBlurKernel,
} from '../../lib/cnn/convolution';
import { Button } from '../ui/Button';

interface KernelEditorProps {
  kernel: Matrix2D;
  onChange: (kernel: Matrix2D) => void;
}

export const KernelEditor: React.FC<KernelEditorProps> = ({ kernel, onChange }) => {
  const [selectedPreset, setSelectedPreset] = useState<string>('custom');

  const presets = {
    edge: createEdgeDetectionKernel(),
    verticalEdge: createVerticalEdgeKernel(),
    horizontalEdge: createHorizontalEdgeKernel(),
    sharpen: createSharpenKernel(),
    blur: createBlurKernel(),
  };

  const handleCellChange = (i: number, j: number, value: string) => {
    const numValue = parseFloat(value) || 0;
    const newKernel = kernel.map((row, ri) =>
      row.map((cell, ci) => (ri === i && ci === j ? numValue : cell))
    );
    onChange(newKernel);
    setSelectedPreset('custom');
  };

  const loadPreset = (presetName: keyof typeof presets) => {
    onChange(presets[presetName]);
    setSelectedPreset(presetName);
  };

  return (
    <div className="space-y-6">
      <div>
        <h4 className="text-lg font-semibold text-gray-800 mb-3">Kernel Editor</h4>
        <div className="inline-grid gap-2" style={{ gridTemplateColumns: `repeat(${kernel[0].length}, 1fr)` }}>
          {kernel.map((row, i) =>
            row.map((cell, j) => (
              <input
                key={`${i}-${j}`}
                type="number"
                step="0.1"
                value={cell.toFixed(2)}
                onChange={(e) => handleCellChange(i, j, e.target.value)}
                className="w-16 h-16 text-center border-2 border-gray-300 rounded focus:border-blue-500 focus:outline-none text-sm font-mono"
              />
            ))
          )}
        </div>
      </div>

      <div>
        <h5 className="text-sm font-semibold text-gray-700 mb-2">Presets</h5>
        <div className="flex flex-wrap gap-2">
          <Button
            size="sm"
            variant={selectedPreset === 'edge' ? 'primary' : 'outline'}
            onClick={() => loadPreset('edge')}
          >
            Edge Detection
          </Button>
          <Button
            size="sm"
            variant={selectedPreset === 'verticalEdge' ? 'primary' : 'outline'}
            onClick={() => loadPreset('verticalEdge')}
          >
            Vertical Edge
          </Button>
          <Button
            size="sm"
            variant={selectedPreset === 'horizontalEdge' ? 'primary' : 'outline'}
            onClick={() => loadPreset('horizontalEdge')}
          >
            Horizontal Edge
          </Button>
          <Button
            size="sm"
            variant={selectedPreset === 'sharpen' ? 'primary' : 'outline'}
            onClick={() => loadPreset('sharpen')}
          >
            Sharpen
          </Button>
          <Button
            size="sm"
            variant={selectedPreset === 'blur' ? 'primary' : 'outline'}
            onClick={() => loadPreset('blur')}
          >
            Blur
          </Button>
        </div>
      </div>
    </div>
  );
};
