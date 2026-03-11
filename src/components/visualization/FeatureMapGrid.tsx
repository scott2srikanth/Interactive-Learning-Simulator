import React, { useEffect, useRef } from 'react';
import { Matrix2D, Matrix3D } from '../../types/cnn';

interface FeatureMapGridProps {
  data: Matrix3D | Matrix2D;
  title?: string;
  maxMaps?: number;
}

export const FeatureMapGrid: React.FC<FeatureMapGridProps> = ({
  data,
  title,
  maxMaps = 16,
}) => {
  const is3D = Array.isArray(data[0][0]);
  const maps: Matrix2D[] = is3D ? (data as Matrix3D).slice(0, maxMaps) : [data as Matrix2D];

  return (
    <div className="space-y-4">
      {title && <h4 className="text-lg font-semibold text-gray-800">{title}</h4>}
      <div className="grid grid-cols-4 gap-4">
        {maps.map((map, idx) => (
          <FeatureMapCanvas key={idx} data={map} label={`Filter ${idx + 1}`} />
        ))}
      </div>
    </div>
  );
};

interface FeatureMapCanvasProps {
  data: Matrix2D;
  label: string;
}

const FeatureMapCanvas: React.FC<FeatureMapCanvasProps> = ({ data, label }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data || data.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const height = data.length;
    const width = data[0].length;

    canvas.width = width;
    canvas.height = height;

    let min = Infinity;
    let max = -Infinity;

    for (let i = 0; i < height; i++) {
      for (let j = 0; j < width; j++) {
        min = Math.min(min, data[i][j]);
        max = Math.max(max, data[i][j]);
      }
    }

    const range = max - min || 1;

    const imageData = ctx.createImageData(width, height);

    for (let i = 0; i < height; i++) {
      for (let j = 0; j < width; j++) {
        const normalized = (data[i][j] - min) / range;
        const value = Math.floor(normalized * 255);

        const idx = (i * width + j) * 4;
        imageData.data[idx] = value;
        imageData.data[idx + 1] = value;
        imageData.data[idx + 2] = value;
        imageData.data[idx + 3] = 255;
      }
    }

    ctx.putImageData(imageData, 0, 0);
  }, [data]);

  return (
    <div className="flex flex-col items-center space-y-2">
      <canvas
        ref={canvasRef}
        className="border border-gray-300 rounded"
        style={{ width: '100px', height: '100px', imageRendering: 'pixelated' }}
      />
      <span className="text-xs text-gray-600">{label}</span>
    </div>
  );
};

interface RGBImageCanvasProps {
  data: Matrix3D;
  label?: string;
}

const RGBImageCanvas: React.FC<RGBImageCanvasProps> = ({ data, label }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data || data.length !== 3) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const height = data[0].length;
    const width = data[0][0].length;

    canvas.width = width;
    canvas.height = height;

    const imageData = ctx.createImageData(width, height);

    for (let i = 0; i < height; i++) {
      for (let j = 0; j < width; j++) {
        const idx = (i * width + j) * 4;
        imageData.data[idx] = Math.floor(data[0][i][j] * 255);
        imageData.data[idx + 1] = Math.floor(data[1][i][j] * 255);
        imageData.data[idx + 2] = Math.floor(data[2][i][j] * 255);
        imageData.data[idx + 3] = 255;
      }
    }

    ctx.putImageData(imageData, 0, 0);
  }, [data]);

  return (
    <div className="flex flex-col items-center space-y-2">
      <canvas
        ref={canvasRef}
        className="border border-gray-300 rounded"
        style={{ width: '150px', height: '150px', imageRendering: 'pixelated' }}
      />
      {label && <span className="text-xs text-gray-600">{label}</span>}
    </div>
  );
};

export { RGBImageCanvas };
