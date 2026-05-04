import { PredictionResult } from '../types/prediction';

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:5000";

/* -----------------------------
   Shared response handler
------------------------------ */
async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let message = `Server error: ${res.status}`;

    try {
      const payload = await res.json();
      if (payload?.error) message = payload.error;
    } catch {
      // ignore non-json responses
    }

    throw new Error(message);
  }

  return res.json();
}

async function prepareImage(blob: Blob): Promise<Blob> {
  return blob;
}

/* -----------------------------
   PREDICTION API
------------------------------ */
export async function predictImage(blob: Blob): Promise<PredictionResult> {
  const formData = new FormData();

  const processedBlob = await prepareImage(blob);

  formData.append("file", processedBlob, "capture.jpg");

  const res = await fetch(`${API_BASE}/api/v1/predict`, {
    method: "POST",
    body: formData,
  });

  return handleResponse<PredictionResult>(res);
}

/* -----------------------------
   DETECTION TYPES
------------------------------ */
export interface OtherPrediction {
  class_id: number;
  class_name: string;
  confidence: number;
}

export interface Detection {
  bbox: [number, number, number, number];
  class_name: string;
  category: string;
  detection_confidence: number;
  classification_confidence: number;
  other_predictions?: OtherPrediction[];
}

export interface DetectionResult {
  detections: Detection[];
}

/* -----------------------------
   DETECTION API
------------------------------ */
export async function detectImage(blob: Blob): Promise<DetectionResult> {
  const formData = new FormData();

  const processedBlob = await prepareImage(blob);

  formData.append("file", processedBlob, "frame.jpg");

  const res = await fetch(`${API_BASE}/api/v1/detect`, {
    method: "POST",
    body: formData,
  });

  return handleResponse<DetectionResult>(res);
}

/* -----------------------------
   LABELS API
------------------------------ */
export interface Label {
  class_id: number;
  class_name: string;
  category: 'prohibitory' | 'warning' | 'mandatory' | 'informational';
  instruction: string;
}

export interface LabelsResponse {
  version: number;
  classes: Label[];
}

let cachedLabels: LabelsResponse | null = null;

export async function fetchLabels(): Promise<LabelsResponse> {
  if (cachedLabels) {
    return cachedLabels;
  }

  const res = await fetch(`${API_BASE}/app/ml/labels.json`);
  cachedLabels = await handleResponse<LabelsResponse>(res);
  return cachedLabels;
}

/**
 * Get instruction for a traffic sign by class name
 */
export async function getInstructionForSign(className: string): Promise<string | null> {
  const labels = await fetchLabels();
  const label = labels.classes.find((l) => l.class_name === className);
  return label?.instruction ?? null;
}