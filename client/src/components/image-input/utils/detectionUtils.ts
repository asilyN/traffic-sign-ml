import { detectImage, Detection } from '@/lib/api';
import { drawBoxes } from './canvasUtils';

export async function performDetection(
  video: HTMLVideoElement,
  captureCanvas: HTMLCanvasElement,
  canvas: HTMLCanvasElement
): Promise<Detection[]> {
  if (video.videoWidth === 0 || video.videoHeight === 0) {
    return [];
  }

  captureCanvas.width = video.videoWidth;
  captureCanvas.height = video.videoHeight;
  captureCanvas.getContext('2d')?.drawImage(video, 0, 0);

  return new Promise((resolve) => {
    captureCanvas.toBlob(
      async (blob) => {
        if (!blob) {
          resolve([]);
          return;
        }
        try {
          const data = await detectImage(blob);
          const dets: Detection[] = data.detections ?? [];
          drawBoxes(canvas, video, dets);
          resolve(dets);
        } catch (err) {
          console.error('Detection failed:', err);
          resolve([]);
        }
      },
      'image/jpeg',
      0.85
    );
  });
}

export async function captureFrame(
  video: HTMLVideoElement,
  captureCanvas: HTMLCanvasElement,
  quality: number = 0.92
): Promise<{ blob: Blob; file: File } | null> {
  if (!video) return null;

  captureCanvas.width = video.videoWidth;
  captureCanvas.height = video.videoHeight;
  captureCanvas.getContext('2d')?.drawImage(video, 0, 0);

  return new Promise((resolve) => {
    captureCanvas.toBlob(
      (blob) => {
        if (!blob) {
          resolve(null);
          return;
        }
        const file = new File([blob], 'frame.jpg', { type: 'image/jpeg' });
        resolve({ blob, file });
      },
      'image/jpeg',
      quality
    );
  });
}
