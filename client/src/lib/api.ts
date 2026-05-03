import { PredictionResult } from '../types/prediction';

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:5000';
const REQUEST_TIMEOUT = 30000; // 30 seconds timeout

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

/**
 * Prediction with timeout and abort support
 */
export async function predictImage(blob: Blob, signal?: AbortSignal): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append('file', blob, 'capture.jpg');

  // Create abort controller with timeout
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);

  // Use provided signal or the one from timeout
  const finalSignal = signal || controller.signal;

  try {
    const res = await fetch(`${API_BASE}/api/v1/predict`, {
      method: 'POST',
      body: formData,
      signal: finalSignal,
    });

    if (!res.ok) {
      throw new Error(await readErrorMessage(res));
    }

    return res.json();
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      throw new Error('Request timeout. The server took too long to respond. Please try again.');
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }
}
