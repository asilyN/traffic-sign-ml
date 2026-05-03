'use client';

import { useRef, useEffect, useState, useCallback } from 'react';
import { Camera, X, Check, RefreshCw } from 'lucide-react';

interface CameraModalProps {
  isOpen: boolean;
  onClose: () => void;
  onCapture: (file: File) => void;
}

export function CameraModal({ isOpen, onClose, onCapture }: CameraModalProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hasCamera, setHasCamera] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // Request camera access and start stream
  useEffect(() => {
    if (!isOpen) return;

    const startCamera = async () => {
      try {
        setError(null);
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: 'environment', // Use back camera on mobile
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          streamRef.current = stream;
          setHasCamera(true);
          setCameraActive(true);
        }
      } catch (err) {
        setHasCamera(false);
        setCameraActive(false);
        setError(
          err instanceof Error ? err.message : 'Unable to access camera. Please check permissions.'
        );
      }
    };

    startCamera();

    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        setCameraActive(false);
      }
    };
  }, [isOpen]);

  const handleCapture = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return;

    const context = canvasRef.current.getContext('2d');
    if (!context) return;

    // Set canvas dimensions to match video
    canvasRef.current.width = videoRef.current.videoWidth;
    canvasRef.current.height = videoRef.current.videoHeight;

    // Draw video frame to canvas
    context.drawImage(videoRef.current, 0, 0);

    // Convert canvas to image
    const imageData = canvasRef.current.toDataURL('image/jpeg');
    setCapturedImage(imageData);

    // Stop camera stream
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      setCameraActive(false);
    }
  }, []);

  const handleRetake = useCallback(() => {
    setCapturedImage(null);
    // Restart camera
    if (videoRef.current) {
      const startCamera = async () => {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({
            video: {
              facingMode: 'environment',
              width: { ideal: 1280 },
              height: { ideal: 720 },
            },
            audio: false,
          });
          videoRef.current!.srcObject = stream;
          streamRef.current = stream;
          setCameraActive(true);
        } catch (err) {
          setError('Failed to restart camera');
        }
      };
      startCamera();
    }
  }, []);

  const handleConfirm = useCallback(() => {
    if (!canvasRef.current) return;

    canvasRef.current.toBlob((blob) => {
      if (!blob) return;

      const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
      onCapture(file);
      setCapturedImage(null);
      setCameraActive(false);
      onClose();
    }, 'image/jpeg');
  }, [onCapture, onClose]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-10 flex items-center justify-center z-50 p-4 backdrop-blur-sm">
      <div className="bg-white rounded-lg shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-[#111827]">Take a Photo</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600 transition-colors">
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Camera or Error */}
        <div className="relative bg-black">
          {error ? (
            <div className="aspect-video flex flex-col items-center justify-center">
              <Camera className="w-16 h-16 text-gray-400 mb-4" />
              <p className="text-white text-center max-w-sm">{error}</p>
              <button
                onClick={onClose}
                className="mt-6 px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 transition-colors"
              >
                Close
              </button>
            </div>
          ) : capturedImage ? (
            <div className="aspect-video flex items-center justify-center">
              <img src={capturedImage} alt="Captured" className="w-full h-full object-contain" />
            </div>
          ) : (
            <video
              ref={videoRef}
              autoPlay
              playsInline
              className="w-full aspect-video object-cover"
            />
          )}
        </div>

        {/* Hidden canvas for capture */}
        <canvas ref={canvasRef} className="hidden" />

        {/* Footer Actions */}
        <div className="flex gap-3 p-4 border-t border-gray-200 bg-gray-50">
          {capturedImage ? (
            <>
              <button
                onClick={handleRetake}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-3 bg-white text-[#111827] border-2 border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
              >
                <RefreshCw className="w-5 h-5" />
                Retake
              </button>
              <button
                onClick={handleConfirm}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-3 bg-[#F97316] text-white rounded-lg hover:bg-[#EA580C] transition-colors"
              >
                <Check className="w-5 h-5" />
                Confirm
              </button>
            </>
          ) : hasCamera && cameraActive ? (
            <>
              <button
                onClick={onClose}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-3 bg-white text-[#111827] border-2 border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleCapture}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-3 bg-[#F97316] text-white rounded-lg hover:bg-[#EA580C] transition-colors"
              >
                <Camera className="w-5 h-5" />
                Capture
              </button>
            </>
          ) : null}
        </div>
      </div>
    </div>
  );
}
