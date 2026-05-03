import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Compress image file to reduce memory and bandwidth usage.
 * Resizes to max 1024px and compresses to 80% JPEG quality.
 * @param file - Original image file
 * @returns Compressed image as Blob
 */
export async function compressImage(file: File): Promise<Blob> {
  const MAX_DIMENSION = 1024;
  const JPEG_QUALITY = 0.8;

  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = (e) => {
      const img = new Image();

      img.onload = () => {
        // Calculate scaled dimensions
        let width = img.width;
        let height = img.height;

        if (width > height) {
          if (width > MAX_DIMENSION) {
            height = Math.round((height * MAX_DIMENSION) / width);
            width = MAX_DIMENSION;
          }
        } else {
          if (height > MAX_DIMENSION) {
            width = Math.round((width * MAX_DIMENSION) / height);
            height = MAX_DIMENSION;
          }
        }

        // Create canvas and draw resized image
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');

        if (!ctx) {
          reject(new Error('Failed to get canvas context'));
          return;
        }

        ctx.drawImage(img, 0, 0, width, height);

        // Convert to JPEG blob with compression
        canvas.toBlob(
          (blob) => {
            if (blob) {
              resolve(blob);
            } else {
              reject(new Error('Failed to compress image'));
            }
          },
          'image/jpeg',
          JPEG_QUALITY
        );
      };

      img.onerror = () => {
        reject(new Error('Failed to load image'));
      };

      img.src = e.target?.result as string;
    };

    reader.onerror = () => {
      reject(new Error('Failed to read file'));
    };

    reader.readAsDataURL(file);
  });
}

/**
 * Preprocess image for CNN inference (matches server-side preprocessing).
 * Converts to RGB, resizes to 32x32, applies ImageNet normalization.
 * Returns normalized pixel data as Uint8Array (visualization) and Float32Array (inference).
 * @param file - Original image file
 * @returns Object with visualization (0-255) and normalized (for model) pixel data
 */
export async function preprocessImageForCNN(file: File): Promise<{
  visualization: Uint8Array; // 0-255 values for display
  normalized: Float32Array; // ImageNet normalized values for model
  width: number;
  height: number;
}> {
  const TARGET_SIZE = 32;
  const IMAGENET_MEAN = [0.485, 0.456, 0.406];
  const IMAGENET_STD = [0.229, 0.224, 0.225];

  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = (e) => {
      const img = new Image();

      img.onload = () => {
        try {
          // Create canvas for resizing
          const canvas = document.createElement('canvas');
          canvas.width = TARGET_SIZE;
          canvas.height = TARGET_SIZE;
          const ctx = canvas.getContext('2d');

          if (!ctx) {
            reject(new Error('Failed to get canvas context for preprocessing'));
            return;
          }

          // Draw resized image
          ctx.drawImage(img, 0, 0, TARGET_SIZE, TARGET_SIZE);

          // Get pixel data
          const imageData = ctx.getImageData(0, 0, TARGET_SIZE, TARGET_SIZE);
          const data = imageData.data; // RGBA values

          // Create output arrays
          const visualization = new Uint8Array(TARGET_SIZE * TARGET_SIZE * 3);
          const normalized = new Float32Array(TARGET_SIZE * TARGET_SIZE * 3);

          // Process each pixel
          for (let i = 0; i < data.length; i += 4) {
            const pixelIndex = i / 4;
            const r = data[i] / 255.0;
            const g = data[i + 1] / 255.0;
            const b = data[i + 2] / 255.0;

            // Store visualization (0-255 RGB)
            visualization[pixelIndex * 3] = data[i];
            visualization[pixelIndex * 3 + 1] = data[i + 1];
            visualization[pixelIndex * 3 + 2] = data[i + 2];

            // Store normalized values using ImageNet normalization
            normalized[pixelIndex * 3] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
            normalized[pixelIndex * 3 + 1] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
            normalized[pixelIndex * 3 + 2] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
          }

          resolve({
            visualization,
            normalized,
            width: TARGET_SIZE,
            height: TARGET_SIZE,
          });
        } catch (error) {
          reject(error);
        }
      };

      img.onerror = () => {
        reject(new Error('Failed to load image for preprocessing'));
      };

      img.src = e.target?.result as string;
    };

    reader.onerror = () => {
      reject(new Error('Failed to read file for preprocessing'));
    };

    reader.readAsDataURL(file);
  });
}

/**
 * Generate a small thumbnail for history display.
 * Much smaller than compressed image - typically 10-20KB as DataURL.
 * @param file - Original image file
 * @returns Thumbnail as DataURL string
 */
export async function generateThumbnail(file: File): Promise<string> {
  const THUMB_SIZE = 200; // 200x200px for history
  const JPEG_QUALITY = 0.6; // Lower quality for thumbnail

  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = (e) => {
      const img = new Image();

      img.onload = () => {
        // Calculate dimensions (keep aspect ratio)
        let width = img.width;
        let height = img.height;

        if (width > height) {
          height = Math.round((height * THUMB_SIZE) / width);
          width = THUMB_SIZE;
        } else {
          width = Math.round((width * THUMB_SIZE) / height);
          height = THUMB_SIZE;
        }

        // Create canvas
        const canvas = document.createElement('canvas');
        canvas.width = THUMB_SIZE;
        canvas.height = THUMB_SIZE;
        const ctx = canvas.getContext('2d');

        if (!ctx) {
          reject(new Error('Failed to create thumbnail'));
          return;
        }

        // Fill background and draw centered image
        ctx.fillStyle = '#f3f4f6';
        ctx.fillRect(0, 0, THUMB_SIZE, THUMB_SIZE);
        const x = (THUMB_SIZE - width) / 2;
        const y = (THUMB_SIZE - height) / 2;
        ctx.drawImage(img, x, y, width, height);

        // Convert to DataURL
        resolve(canvas.toDataURL('image/jpeg', JPEG_QUALITY));
      };

      img.onerror = () => {
        reject(new Error('Failed to load image for thumbnail'));
      };

      img.src = e.target?.result as string;
    };

    reader.onerror = () => {
      reject(new Error('Failed to read file'));
    };

    reader.readAsDataURL(file);
  });
}
