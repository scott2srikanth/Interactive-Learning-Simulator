import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Matrix2D, Matrix3D } from '../../types/cnn';
import { Play, Pause, RotateCcw, SkipForward } from 'lucide-react';
import { Button } from '../ui/Button';

interface ConvolutionAnimationProps {
  input: Matrix2D;
  kernel: Matrix2D;
  output: Matrix2D;
  stride?: number;
}

export const ConvolutionAnimation: React.FC<ConvolutionAnimationProps> = ({
  input,
  kernel,
  output,
  stride = 1,
}) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentPosition, setCurrentPosition] = useState({ row: 0, col: 0 });
  const [showCalculation, setShowCalculation] = useState(false);

  const kernelSize = kernel.length;
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
        setShowCalculation(true);
        setTimeout(() => setShowCalculation(false), 800);
      }, 1500);

      return () => clearTimeout(timer);
    }
  }, [isPlaying, currentPosition, outputHeight, outputWidth]);

  const handleReset = () => {
    setCurrentPosition({ row: 0, col: 0 });
    setIsPlaying(false);
    setShowCalculation(false);
  };

  const handleNext = () => {
    const { row, col } = currentPosition;
    const nextCol = col + 1;
    const nextRow = nextCol >= outputWidth ? row + 1 : row;
    const newCol = nextCol >= outputWidth ? 0 : nextCol;

    if (nextRow < outputHeight) {
      setCurrentPosition({ row: nextRow, col: newCol });
      setShowCalculation(true);
      setTimeout(() => setShowCalculation(false), 800);
    }
  };

  const handleFastForward = () => {
    setCurrentPosition({ row: outputHeight - 1, col: outputWidth - 1 });
    setIsPlaying(false);
    setShowCalculation(false);
  };

  const inputRow = currentPosition.row * stride;
  const inputCol = currentPosition.col * stride;

  const getInputValue = (r: number, c: number): number => {
    return input[r]?.[c] ?? 0;
  };

  const isInKernelRegion = (r: number, c: number): boolean => {
    return (
      r >= inputRow &&
      r < inputRow + kernelSize &&
      c >= inputCol &&
      c < inputCol + kernelSize
    );
  };

  const calculateConvolution = (): { steps: string[]; result: number } => {
    let sum = 0;
    const steps: string[] = [];

    for (let kr = 0; kr < kernelSize; kr++) {
      for (let kc = 0; kc < kernelSize; kc++) {
        const inputVal = getInputValue(inputRow + kr, inputCol + kc);
        const kernelVal = kernel[kr][kc];
        const product = inputVal * kernelVal;
        sum += product;
        steps.push(`${inputVal.toFixed(2)} × ${kernelVal.toFixed(2)} = ${product.toFixed(2)}`);
      }
    }

    return { steps, result: sum };
  };

  const { steps, result } = calculateConvolution();

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
          <h4 className="text-lg font-bold text-gray-900">Convolution Operation</h4>
          <p className="text-sm text-gray-600">
            Position: ({currentPosition.row}, {currentPosition.col}) • Output: {result.toFixed(3)}
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

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="space-y-3">
          <h5 className="text-sm font-semibold text-gray-700">Input Feature Map</h5>
          <div className="bg-white p-4 rounded-lg border-2 border-gray-300 relative overflow-hidden">
            <div
              className={`grid ${getGap(inputSize)}`}
              style={{ gridTemplateColumns: `repeat(${input[0].length}, 1fr)` }}
            >
              {input.map((row, r) =>
                row.map((val, c) => (
                  <motion.div
                    key={`${r}-${c}`}
                    className={`aspect-square flex items-center justify-center ${getFontSize(inputSize)} font-mono transition-all ${
                      isInKernelRegion(r, c)
                        ? 'bg-blue-500 text-white border border-blue-700 z-10 shadow-lg'
                        : 'text-gray-700 border border-gray-200'
                    }`}
                    style={{
                      backgroundColor: isInKernelRegion(r, c)
                        ? undefined
                        : `rgb(${Math.floor((1 - val) * 255)}, ${Math.floor((1 - val) * 255)}, ${Math.floor((1 - val) * 255)})`
                    }}
                    animate={
                      isInKernelRegion(r, c)
                        ? { scale: [1, 1.05, 1.05], backgroundColor: ['#3b82f6', '#2563eb', '#3b82f6'] }
                        : { scale: 1 }
                    }
                    transition={{ duration: 0.5 }}
                  >
                    {showValues ? val.toFixed(1) : ''}
                  </motion.div>
                ))
              )}
            </div>
            <motion.div
              className="absolute border-4 border-yellow-400 pointer-events-none shadow-2xl rounded-sm bg-yellow-500 bg-opacity-10"
              animate={{
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                transform: `translate(${(inputCol / input[0].length) * 100}%, ${(inputRow / input.length) * 100}%) scale(${kernelSize / input[0].length}, ${kernelSize / input.length})`,
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
          <h5 className="text-sm font-semibold text-gray-700">Kernel (Filter)</h5>
          <div className="bg-gradient-to-br from-yellow-50 to-orange-50 p-4 rounded-lg border-2 border-yellow-400">
            <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${kernelSize}, 1fr)` }}>
              {kernel.map((row, r) =>
                row.map((val, c) => (
                  <motion.div
                    key={`${r}-${c}`}
                    className="aspect-square flex items-center justify-center text-sm font-bold bg-yellow-200 text-gray-900 rounded border-2 border-yellow-400"
                    animate={isPlaying ? { scale: [1, 1.1, 1] } : {}}
                    transition={{ duration: 0.5, repeat: Infinity, repeatDelay: 1 }}
                  >
                    {val.toFixed(1)}
                  </motion.div>
                ))
              )}
            </div>
            <p className="text-xs text-center mt-2 text-gray-700 font-semibold">
              {kernelSize}×{kernelSize} Kernel
            </p>
          </div>

          <AnimatePresence>
            {showCalculation && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 10 }}
                className="bg-gray-900 text-gray-100 p-3 rounded-lg text-xs font-mono max-h-32 overflow-y-auto"
              >
                <p className="text-green-400 mb-1">Element-wise Multiplication:</p>
                {steps.map((step, idx) => (
                  <motion.p
                    key={idx}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: idx * 0.05 }}
                  >
                    {step}
                  </motion.p>
                ))}
                <p className="text-yellow-400 mt-2 font-bold border-t border-gray-700 pt-1">
                  Sum = {result.toFixed(3)}
                </p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <div className="space-y-3">
          <h5 className="text-sm font-semibold text-gray-700">Output Feature Map</h5>
          <div className="bg-green-50 p-4 rounded-lg border-2 border-green-300">
            <div
              className={`grid ${getGap(outputHeight)}`}
              style={{ gridTemplateColumns: `repeat(${outputWidth}, 1fr)` }}
            >
              {output.map((row, r) =>
                row.map((val, c) => {
                  const isCurrentCell = r === currentPosition.row && c === currentPosition.col;
                  const isProcessed = r * outputWidth + c < currentPosition.row * outputWidth + currentPosition.col;

                  const normalizedVal = Math.abs(val);
                  const intensity = Math.min(normalizedVal / 5, 1);
                  const grayValue = Math.floor((1 - intensity) * 200 + 55);

                  return (
                    <motion.div
                      key={`${r}-${c}`}
                      className={`aspect-square flex items-center justify-center ${getFontSize(outputHeight)} font-mono transition-all border relative ${
                        isCurrentCell
                          ? 'bg-green-500 text-white border-green-700 z-20 shadow-2xl font-bold ring-4 ring-green-300'
                          : isProcessed
                          ? 'text-gray-800 border-gray-300'
                          : 'text-gray-400 border-gray-200'
                      }`}
                      style={{
                        backgroundColor: isCurrentCell ? undefined : isProcessed
                          ? `rgb(${grayValue}, ${grayValue}, ${grayValue})`
                          : '#f9fafb'
                      }}
                      animate={
                        isCurrentCell
                          ? {
                              scale: [1, 1.15, 1.1],
                              rotate: [0, 2, -2, 0],
                              boxShadow: ['0px 0px 0px rgba(34, 197, 94, 0)', '0px 0px 20px rgba(34, 197, 94, 0.8)', '0px 0px 15px rgba(34, 197, 94, 0.6)']
                            }
                          : {}
                      }
                      transition={{ duration: 0.4 }}
                    >
                      {isCurrentCell && (
                        <motion.div
                          className="absolute inset-0 bg-green-400 rounded-sm"
                          initial={{ opacity: 0 }}
                          animate={{ opacity: [0, 0.5, 0] }}
                          transition={{ duration: 1, repeat: Infinity }}
                        />
                      )}
                      <span className="relative z-10">
                        {(isProcessed || isCurrentCell) && showValues ? val.toFixed(1) : ''}
                      </span>
                    </motion.div>
                  );
                })
              )}
            </div>
          </div>
          {!showValues && (
            <p className="text-xs text-gray-500 italic">
              Values hidden for clarity (grid too large)
            </p>
          )}
        </div>
      </div>

      <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded">
        <p className="text-sm text-gray-800">
          <strong>How it works:</strong> The {kernelSize}×{kernelSize} kernel slides across the input, computing
          element-wise multiplication and summing the results. The yellow box shows the current region being
          processed, and the output value appears in the green grid.
        </p>
      </div>
    </div>
  );
};
