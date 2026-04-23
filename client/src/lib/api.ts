import { PredictionResult } from "../types/prediction";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:5000";

export async function predictImage(blob: Blob): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append("file", blob, "capture.jpg");

  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error(`Server error: ${res.status}`);
  }

  return res.json();
}