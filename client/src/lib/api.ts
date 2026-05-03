import { PredictionResult } from '../types/prediction';

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:5000";

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let message = `Server error: ${res.status}`;
    try {
      const payload = (await res.json()) as { error?: string };
      if (payload.error) message = payload.error;
    } catch {
      // Keep generic message when body is not JSON.
    }
    throw new Error(message);
  }
  return res.json();
}

export async function predictImage(blob: Blob): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append("file", blob, "capture.jpg");

  const res = await fetch(`${API_BASE}/api/v1/predict`, {
    method: "POST",
    body: formData,
  });

  return handleResponse<PredictionResult>(res);
}

export interface Detection {
  bbox: [number, number, number, number];
  class_name: string;
  category: string;
  detection_confidence: number;
  classification_confidence: number;
}

export interface DetectionResult {
  detections: Detection[];
}

export async function detectImage(blob: Blob): Promise<DetectionResult> {
  const formData = new FormData();
  formData.append("file", blob, "frame.jpg");

  const res = await fetch(`${API_BASE}/api/v1/detect`, {
    method: "POST",
    body: formData,
  });

  return handleResponse<DetectionResult>(res);
}