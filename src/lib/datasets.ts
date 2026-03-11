import { Matrix3D } from '../types/cnn';
import { create3DMatrix } from './cnn/matrix';

export interface DatasetImage {
  id: string;
  name: string;
  data: Matrix3D;
  label: string;
}

export function createSampleImage(pattern: 'vertical' | 'horizontal' | 'diagonal' | 'circle'): Matrix3D {
  const size = 28;
  const image = create3DMatrix(1, size, size, 0);

  switch (pattern) {
    case 'vertical':
      for (let i = 0; i < size; i++) {
        for (let j = 12; j < 16; j++) {
          image[0][i][j] = 1;
        }
      }
      break;

    case 'horizontal':
      for (let i = 12; i < 16; i++) {
        for (let j = 0; j < size; j++) {
          image[0][i][j] = 1;
        }
      }
      break;

    case 'diagonal':
      for (let i = 0; i < size; i++) {
        const j = i;
        if (j < size) {
          image[0][i][j] = 1;
          if (j > 0) image[0][i][j - 1] = 0.5;
          if (j < size - 1) image[0][i][j + 1] = 0.5;
        }
      }
      break;

    case 'circle':
      const centerX = size / 2;
      const centerY = size / 2;
      const radius = 8;
      for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
          const dist = Math.sqrt((i - centerX) ** 2 + (j - centerY) ** 2);
          if (Math.abs(dist - radius) < 2) {
            image[0][i][j] = 1;
          }
        }
      }
      break;
  }

  return image;
}

export function createDigitImage(digit: number): Matrix3D {
  const size = 28;
  const image = create3DMatrix(1, size, size, 0);

  const patterns: { [key: number]: (img: Matrix3D) => void } = {
    0: (img) => {
      for (let i = 5; i < 23; i++) {
        for (let j = 8; j < 20; j++) {
          const distFromCenter = Math.sqrt((i - 14) ** 2 + (j - 14) ** 2);
          if (distFromCenter > 5 && distFromCenter < 8) {
            img[0][i][j] = 1;
          }
        }
      }
    },
    1: (img) => {
      for (let i = 5; i < 23; i++) {
        img[0][i][14] = 1;
        img[0][i][13] = 0.8;
      }
    },
    2: (img) => {
      for (let j = 8; j < 20; j++) {
        img[0][7][j] = 1;
        img[0][14][j] = 1;
        img[0][21][j] = 1;
      }
      for (let i = 7; i < 14; i++) {
        img[0][i][19] = 1;
      }
      for (let i = 14; i < 21; i++) {
        img[0][i][8] = 1;
      }
    },
  };

  if (patterns[digit]) {
    patterns[digit](image);
  }

  return image;
}

export function createColorImage(pattern: 'red-square' | 'blue-circle' | 'green-triangle'): Matrix3D {
  const size = 28;
  const image = create3DMatrix(3, size, size, 0);

  switch (pattern) {
    case 'red-square':
      for (let i = 8; i < 20; i++) {
        for (let j = 8; j < 20; j++) {
          image[0][i][j] = 1;
          image[1][i][j] = 0;
          image[2][i][j] = 0;
        }
      }
      break;

    case 'blue-circle':
      const centerX = size / 2;
      const centerY = size / 2;
      const radius = 8;
      for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
          const dist = Math.sqrt((i - centerX) ** 2 + (j - centerY) ** 2);
          if (dist < radius) {
            image[0][i][j] = 0;
            image[1][i][j] = 0.5;
            image[2][i][j] = 1;
          }
        }
      }
      break;

    case 'green-triangle':
      for (let i = 8; i < 20; i++) {
        const width = (i - 8) * 1.5;
        const left = Math.floor(14 - width / 2);
        const right = Math.floor(14 + width / 2);
        for (let j = left; j <= right; j++) {
          if (j >= 0 && j < size) {
            image[0][i][j] = 0;
            image[1][i][j] = 0.8;
            image[2][i][j] = 0.2;
          }
        }
      }
      break;
  }

  return image;
}

export const sampleDataset: DatasetImage[] = [
  {
    id: 'vertical-line',
    name: 'Vertical Line',
    data: createSampleImage('vertical'),
    label: 'Line',
  },
  {
    id: 'horizontal-line',
    name: 'Horizontal Line',
    data: createSampleImage('horizontal'),
    label: 'Line',
  },
  {
    id: 'diagonal-line',
    name: 'Diagonal Line',
    data: createSampleImage('diagonal'),
    label: 'Line',
  },
  {
    id: 'circle',
    name: 'Circle',
    data: createSampleImage('circle'),
    label: 'Shape',
  },
  {
    id: 'digit-0',
    name: 'Digit 0',
    data: createDigitImage(0),
    label: '0',
  },
  {
    id: 'digit-1',
    name: 'Digit 1',
    data: createDigitImage(1),
    label: '1',
  },
  {
    id: 'digit-2',
    name: 'Digit 2',
    data: createDigitImage(2),
    label: '2',
  },
  {
    id: 'color-red-square',
    name: 'Red Square (RGB)',
    data: createColorImage('red-square'),
    label: 'Color',
  },
  {
    id: 'color-blue-circle',
    name: 'Blue Circle (RGB)',
    data: createColorImage('blue-circle'),
    label: 'Color',
  },
  {
    id: 'color-green-triangle',
    name: 'Green Triangle (RGB)',
    data: createColorImage('green-triangle'),
    label: 'Color',
  },
];
