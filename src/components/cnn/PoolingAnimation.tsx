import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Matrix2D } from '../../types/cnn';
import { Play, Pause, RotateCcw, SkipForward } from 'lucide-react';
import { Button } from '../ui/Button';

interface PoolingAnimationProps {
  input: Matrix2D;
  output: Matrix2D;
  poolSize: number;
  poolType: 'max' | 'average';
  stride?: number;
}

export const PoolingAnimation: React.FC<PoolingAnimationProps> = ({
  input,
  output,
  poolSize,
  poolType,
  stride = 2,
}) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentPosition, setCurrentPosition] = useState({ row: 0, col: 0 });

  const outputHeight = output.length;
  const outputWidth = output[0]?.length || 0;

  useEffect(() => {
    if (isPlaying) {
      const timer = setTimeout(() => {
        const { row, col } = currentPosition;
        const nextCol = col + 1;
        const nextRow = nextCol >= outputWidth ? row + 1 : row;
        const newCol = nextCol >= outputWidth ? 0 : nextCol;

        if (nextRow >= outputHeight) {
          setIsPlaying(false);
          return;
        }

        setCurrentPosition({ row: nextRow, col: newCol });
      }, 1500);

      return () => clearTimeout(timer);
    }
  }, [isPlaying, currentPosition, outputHeight, outputWidth]);

  const handleReset = () => {
    setCurrentPosition({ row: 0, col: 0 });
    setIsPlaying(false);
  };

  const handleNext = () => {
    const { row, col } = currentPosition;
    const nextCol = col + 1;
    const nextRow = nextCol >= outputWidth ? row + 1 : row;
    const newCol = nextCol >= outputWidth ? 0 : nextCol;

    if (nextRow < outputHeight) {
      setCurrentPosition({ row: nextRow, col: newCol });
    }
  };

  const handleFastForward = () => {
    setCurrentPosition({ row: outputHeight - 1, col: outputWidth - 1 });
    setIsPlaying(false);
  };

  const inputRow = currentPosition.row * stride;
  const inputCol = currentPosition.col * stride;

  const isInPoolRegion = (r: number, c: number): boolean => {
    return (
      r >= inputRow &&
      r < inputRow + poolSize &&
      c >= inputCol &&
      c < inputCol + poolSize
    );
  };

  const calculatePooling = (): { values: number[]; result: number; maxIdx?: number } => {
    const values: number[] = [];
    let maxIdx = 0;

    for (let pr = 0; pr < poolSize; pr++) {
      for (let pc = 0; pc < poolSize; pc++) {
        const val = input[inputRow + pr]?.[inputCol + pc] ?? 0;
        if (val > values[maxIdx]) {
          maxIdx = values.length;
        }
        values.push(val);
      }
    }

    const result = poolType === 'max'
      ? Math.max(...values)
      : values.reduce((a, b) => a + b, 0) / values.length;

    return { values, result, maxIdx: poolType === 'max' ? maxIdx : undefined };
  };

  const { values, result, maxIdx } = calculatePooling();

  const inputSize = input.length;
  const getFontSize = (gridSize: number) => {
    if (gridSize > 20) return 'text-[0.4rem]';
    if (gridSize > 10) return 'text-[0.55rem]';
    return 'text-xs';
  };

  const getGap = (gridSize: number) => {
    if (gridSize > 20) return 'gap-0';
    if (gridSize > 10) return 'gap-0.5';
    return 'gap-1';
  };

  const showValues = inputSize <= 15;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h4 className="text-lg font-bold text-gray-900">
            {poolType === 'max' ? 'Max' : 'Average'} Pooling Operation
          </h4>
          <p className="text-sm text-gray-600">
            Position: ({currentPosition.row}, {currentPosition.col}) • {poolSize}×{poolSize} pool • Output:{' '}
            {result.toFixed(3)}
          </p>
        </div>
        <div className="flex gap-2">
          <Button size="sm" variant="outline" onClick={handleReset} title="Reset to start">
            <RotateCcw className="w-4 h-4" />
          </Button>
          <Button size="sm" variant="outline" onClick={handleNext} title="Next step">
            <SkipForward className="w-4 h-4" />
          </Button>
          <Button size="sm" variant="outline" onClick={handleFastForward} title="Complete animation">
            <SkipForward className="w-4 h-4" />
            <SkipForward className="w-4 h-4 -ml-2" />
          </Button>
          {isPlaying ? (
            <Button size="sm" onClick={() => setIsPlaying(false)}>
              <Pause className="w-4 h-4 mr-1" />
              Pause
            </Button>
          ) : (
            <Button size="sm" onClick={() => setIsPlaying(true)}>
              <Play className="w-4 h-4 mr-1" />
              Play
            </Button>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-3">
          <h5 className="text-sm font-semibold text-gray-700">Input Feature Map</h5>
          <div className="bg-white p-4 rounded-lg border-2 border-gray-300 relative overflow-hidden">
            <div
              className={`grid ${getGap(inputSize)}`}
              style={{ gridTemplateColumns: `repeat(${input[0].length}, 1fr)` }}
            >
              {input.map((row, r) =>
                row.map((val, c) => {
                  const isInRegion = isInPoolRegion(r, c);
                  const regionIdx = isInRegion
                    ? (r - inputRow) * poolSize + (c - inputCol)
                    : -1;
                  const isMax = poolType === 'max' && regionIdx === maxIdx;

                  return (
                    <motion.div
                      key={`${r}-${c}`}
                      className={`aspect-square flex items-center justify-center ${getFontSize(inputSize)} font-mono transition-all border ${
                        isMax
                          ? 'bg-red-500 text-white border-red-700 z-20 shadow-xl'
                          : isInRegion
                          ? 'bg-orange-400 text-white border-orange-600 z-10'
                          : 'text-gray-700 border-gray-200'
                      }`}
                      style={{
                        backgroundColor:
                          isMax || isInRegion
                            ? undefined
                            : `rgb(${Math.floor((1 - val) * 255)}, ${Math.floor((1 - val) * 255)}, ${Math.floor((1 - val) * 255)})`
                      }}
                      animate={
                        isMax
                          ? { scale: [1, 1.15, 1.1], rotate: [0, 5, -5, 0] }
                          : isInRegion
                          ? { scale: [1, 1.05, 1.05] }
                          : { scale: 1 }
                      }
                      transition={{ duration: 0.5 }}
                    >
                      {showValues ? val.toFixed(1) : ''}
                    </motion.div>
                  );
                })
              )}
            </div>
            <motion.div
              className="absolute border-4 border-orange-400 pointer-events-none shadow-2xl rounded-sm bg-orange-500 bg-opacity-10"
              animate={{
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                transform: `translate(${(inputCol / input[0].length) * 100}%, ${(inputRow / input.length) * 100}%) scale(${poolSize / input[0].length}, ${poolSize / input.length})`,
                transformOrigin: 'top left',
              }}
              transition={{ type: 'spring', stiffness: 200, damping: 20 }}
            />
          </div>
          {!showValues && (
            <p className="text-xs text-gray-500 italic">
              Values hidden for clarity (grid too large)
            </p>
          )}
        </div>

        <div className="space-y-3">
          <h5 className="text-sm font-semibold text-gray-700">Output Feature Map</h5>
          <div className="bg-green-50 p-4 rounded-lg border-2 border-green-300">
            <div
              className={`grid ${getGap(outputHeight)}`}
              style={{ gridTemplateColumns: `repeat(${outputWidth}, 1fr)` }}
            >
              {output.map((row, r) =>
                row.map((val, c) => (
                  <motion.div
                    key={`${r}-${c}`}
                    className={`aspect-square flex items-center justify-center ${getFontSize(outputHeight)} font-mono transition-all border ${
                      r === currentPosition.row && c === currentPosition.col
                        ? 'bg-green-600 text-white border-green-800 z-10 shadow-lg'
                        : r * outputWidth + c < currentPosition.row * outputWidth + currentPosition.col
                        ? 'bg-green-400 text-white border-green-500'
                        : 'text-gray-400 border-gray-300'
                    }`}
                    style={{
                      backgroundColor:
                        r === currentPosition.row && c === currentPosition.col
                          ? undefined
                          : r * outputWidth + c < currentPosition.row * outputWidth + currentPosition.col
                          ? undefined
                          : `rgb(${Math.floor((1 - val) * 255)}, ${Math.floor((1 - val) * 255)}, ${Math.floor((1 - val) * 255)})`
                    }}
                    animate={
                      r === currentPosition.row && c === currentPosition.col
                        ? { scale: [1, 1.1, 1.05] }
                        : {}
                    }
                  >
                    {showValues ? val.toFixed(1) : ''}
                  </motion.div>
                ))
              )}
            </div>
          </div>
          {!showValues && (
            <p className="text-xs text-gray-500 italic">
              Values hidden for clarity (grid too large)
            </p>
          )}

          <div className="bg-gray-900 text-gray-100 p-4 rounded-lg text-xs font-mono">
            <p className="text-orange-400 mb-2 font-bold">Values in pool region:</p>
            <div className="grid grid-cols-2 gap-1 mb-2">
              {values.map((val, idx) => (
                <motion.span
                  key={idx}
                  className={`${
                    poolType === 'max' && idx === maxIdx
                      ? 'text-red-400 font-bold'
                      : 'text-gray-300'
                  }`}
                  animate={poolType === 'max' && idx === maxIdx ? { scale: [1, 1.2, 1] } : {}}
                >
                  {val.toFixed(2)}
                  {poolType === 'max' && idx === maxIdx && ' ⭐'}
                </motion.span>
              ))}
            </div>
            <p className="text-green-400 font-bold border-t border-gray-700 pt-2">
              {poolType === 'max' ? 'Max' : 'Average'} = {result.toFixed(3)}
            </p>
          </div>
        </div>
      </div>

      <div className="bg-orange-50 border-l-4 border-orange-500 p-4 rounded">
        <p className="text-sm text-gray-800">
          <strong>How it works:</strong> {poolType === 'max' ? 'Max pooling' : 'Average pooling'} reduces spatial
          dimensions by taking the {poolType === 'max' ? 'maximum' : 'average'} value from each {poolSize}×{poolSize}{' '}
          region. The orange box shows the current pool region
          {poolType === 'max' && ', and the red cell highlights the maximum value'}.
        </p>
      </div>
    </div>
  );
};
