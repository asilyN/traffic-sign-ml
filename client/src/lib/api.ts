import { PredictionResult } from "../types/prediction";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:5000";

async function readErrorMessage(res: Response): Promise<string> {
  let message = `Server error: ${res.status}`;
  try {
    const payload = (await res.json()) as { error?: string };
    if (payload.error) {
      message = payload.error;
    }
  } catch {
    // Keep generic message when body is not JSON.
  }
  return message;
}

export async function predictImage(blob: Blob): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append("file", blob, "capture.jpg");

  const res = await fetch(`${API_BASE}/api/v1/predict`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error(await readErrorMessage(res));
  }

  return res.json();
}
