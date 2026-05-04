export async function startCameraStream(
  videoElement: HTMLVideoElement,
  facingMode: 'environment' | 'user'
): Promise<MediaStream | null> {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: facingMode,
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
    });
    videoElement.srcObject = stream;
    return stream;
  } catch (err) {
    console.error('Failed to get camera stream:', err);
    return null;
  }
}

export function stopCameraStream(stream: MediaStream | null) {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
  }
}
