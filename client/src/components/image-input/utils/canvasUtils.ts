import { Detection } from '@/lib/api';

export function drawBoxes(
  canvas: HTMLCanvasElement,
  video: HTMLVideoElement,
  detections: Detection[]
) {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (const det of detections) {
    const [x1, y1, x2, y2] = det.bbox;
    const label = `${det.class_name} ${(det.classification_confidence * 100).toFixed(0)}%`;

    ctx.strokeStyle = '#F97316';
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    ctx.font = 'bold 13px sans-serif';
    const textW = ctx.measureText(label).width;
    ctx.fillStyle = '#F97316';
    ctx.fillRect(x1, y1 - 20, textW + 8, 20);

    ctx.fillStyle = '#fff';
    ctx.fillText(label, x1 + 4, y1 - 5);
  }
}

export function clearCanvas(canvas: HTMLCanvasElement) {
  const ctx = canvas.getContext('2d');
  if (canvas && ctx) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
}
