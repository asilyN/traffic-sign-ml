'use client';

import { Upload, Loader2, X, FlipHorizontal } from 'lucide-react';
import { useRef, useState, useCallback, useEffect } from 'react';
import { detectImage, Detection } from '@/src/lib/api';

interface ImageInputProps {
  selectedImage: string | null;
  onImageSelect: (imageUrl: string, file: File) => void;
  onDetect: () => void;
  isDetecting: boolean;
  onFrameCapture?: (imageUrl: string, detections: Detection[]) => void;
}

export function ImageInput({
  selectedImage,
  onImageSelect,
  onDetect,
  isDetecting,
  onFrameCapture,
}: ImageInputProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [cameraActive, setCameraActive] = useState(false);
  const [facingMode, setFacingMode] = useState<'environment' | 'user'>('environment');
  const [isScanning, setIsScanning] = useState(false);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [isContinuousDetecting, setIsContinuousDetecting] = useState(false);
  const [hoveredDetectionIndex, setHoveredDetectionIndex] = useState<number | null>(null);
  const detectionIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const isDetectingRef = useRef(false);

  // ── Draw YOLO boxes on canvas overlay ─────────────────────────────────────
  const drawBoxes = useCallback((dets: Detection[]) => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (const det of dets) {
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
  }, []);

  // ── Continuous detection loop for camera ─────────────────────────────────────
  const runContinuousDetection = useCallback(async () => {
    const video = videoRef.current;
    const capture = captureCanvasRef.current;
    if (!video || !capture || isDetectingRef.current) return;

    // Skip if video is not ready
    if (video.videoWidth === 0 || video.videoHeight === 0) return;

    try {
      isDetectingRef.current = true;
      setIsContinuousDetecting(true);

      // Capture frame
      capture.width = video.videoWidth;
      capture.height = video.videoHeight;
      capture.getContext('2d')?.drawImage(video, 0, 0);

      capture.toBlob(async (blob) => {
        if (!blob) return;
        try {
          const data = await detectImage(blob);
          const dets: Detection[] = data.detections ?? [];
          
          // Only update if we have detections or to clear previous ones
          setDetections(dets);
          drawBoxes(dets);
        } catch (err) {
          console.error('Continuous detection failed:', err);
          // Keep previous detections on error
        } finally {
          isDetectingRef.current = false;
          setIsContinuousDetecting(false);
        }
      }, 'image/jpeg', 0.85);
    } catch (err) {
      console.error('Frame capture failed:', err);
      isDetectingRef.current = false;
      setIsContinuousDetecting(false);
    }
  }, [drawBoxes]);

  // ── Start/stop continuous detection ──────────────────────────────────────────
  useEffect(() => {
    if (cameraActive && !selectedImage) {
      // Start continuous detection with 800ms interval
      detectionIntervalRef.current = setInterval(() => {
        runContinuousDetection();
      }, 800);

      return () => {
        if (detectionIntervalRef.current) {
          clearInterval(detectionIntervalRef.current);
          detectionIntervalRef.current = null;
        }
      };
    }
  }, [cameraActive, selectedImage, runContinuousDetection]);

  // ── Cleanup on unmount ───────────────────────────────────────────────────────
  const captureFrameForManualScan = useCallback(async () => {
    const video = videoRef.current;
    const capture = captureCanvasRef.current;
    if (!video || !capture || isScanning) return;

    // Pause continuous detection
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
      detectionIntervalRef.current = null;
    }

    capture.width = video.videoWidth;
    capture.height = video.videoHeight;
    capture.getContext('2d')?.drawImage(video, 0, 0);

    capture.toBlob(async (blob) => {
      if (!blob) return;
      setIsScanning(true);
      try {
        const data = await detectImage(blob);
        const dets: Detection[] = data.detections ?? [];
        
        // Freeze current detections on canvas
        setDetections(dets);
        drawBoxes(dets);

        if (dets.length > 0) {
          // Sort by confidence and get top 3
          const topDets = dets.sort((a, b) => b.classification_confidence - a.classification_confidence).slice(0, 3);

          // Convert blob to data URL for history
          const reader = new FileReader();
          reader.onload = (e) => {
            const imageUrl = e.target?.result as string;
            onFrameCapture?.(imageUrl, topDets);
          };
          reader.readAsDataURL(blob);
        }
      } catch (err) {
        console.error('Manual scan detection failed:', err);
      } finally {
        setIsScanning(false);
      }
    }, 'image/jpeg', 0.92);
  }, [isScanning, onFrameCapture, drawBoxes]);

  // ── Camera start/stop ─────────────────────────────────────────────────────
  const startCamera = useCallback(async (facing: 'environment' | 'user' = facingMode) => {
    setCameraError(null);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: facing, width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      setCameraActive(true);
    } catch (err) {
      setCameraError('Camera access denied or not available.');
      console.error(err);
    }
  }, [facingMode]);

  const stopCamera = useCallback(() => {
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
      detectionIntervalRef.current = null;
    }
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setCameraActive(false);
    setDetections([]);
  }, []);

  // ── Auto-start camera ────────────────────────────────────────────────────────
  useEffect(() => {
    // Only auto-start if no image is selected
    if (!selectedImage) {
      startCamera();
    }
    return () => stopCamera();
  }, [selectedImage, startCamera, stopCamera]);

  const flipCamera = () => {
    const next = facingMode === 'environment' ? 'user' : 'environment';
    setFacingMode(next);
    startCamera(next);
  };

  // ── Upload handler ─────────────────────────────────────────────────────────
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    // Stop camera when image is selected
    stopCamera();
    const reader = new FileReader();
    reader.onload = (event) => onImageSelect(event.target?.result as string, file);
    reader.readAsDataURL(file);
  };

  const handleClearImage = () => {
    // Trigger parent to clear image
    onImageSelect('', new File([], ''));
    setDetections([]);
    // Restart camera will happen automatically via useEffect
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <h2 className="text-[#111827] mb-4">Detection Input</h2>

      {/* Upload button always visible */}
      <div className="mb-6">
        <button
          onClick={() => fileInputRef.current?.click()}
          className="flex items-center justify-center gap-2 px-4 py-3 bg-[#F97316] text-white rounded-lg hover:bg-[#EA580C] transition-colors"
        >
          <Upload className="w-5 h-5" />
          Upload Image
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="hidden"
        />
      </div>

      {/* Image upload mode */}
      {selectedImage && !cameraActive && (
        <div className="space-y-4">
          <div className="relative border-2 border-gray-200 rounded-lg overflow-hidden bg-gray-50">
            <img
              src={selectedImage}
              alt="Selected"
              className="w-full h-auto max-h-96 object-contain"
            />
          </div>
          <button
            onClick={onDetect}
            disabled={isDetecting}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-[#F97316] text-white rounded-lg hover:bg-[#EA580C] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isDetecting ? (
              <><Loader2 className="w-5 h-5 animate-spin" /> Detecting...</>
            ) : (
              'Detect Sign'
            )}
          </button>
          <button
            onClick={handleClearImage}
            className="w-full px-4 py-2 text-[#6B7280] border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
          >
            Back to Camera
          </button>
        </div>
      )}

      {/* Camera mode */}
      {cameraActive && !selectedImage && (
        <div className="space-y-3">
          <div className="relative rounded-lg overflow-hidden bg-black">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full max-h-96 object-contain"
            />
            <canvas
              ref={canvasRef}
              className="absolute inset-0 w-full h-full pointer-events-none"
            />
            <button
              onClick={() => {
                // Clear selected image by calling onImageSelect with empty values
                onImageSelect('', new File([], ''));
                stopCamera();
              }}
              className="absolute top-2 right-2 p-1.5 bg-black/60 text-white rounded-full hover:bg-black/80 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
            <button
              onClick={() => {
                const next = facingMode === 'environment' ? 'user' : 'environment';
                setFacingMode(next);
                startCamera(next);
              }}
              className="absolute top-2 left-2 p-1.5 bg-black/60 text-white rounded-full hover:bg-black/80 transition-colors"
            >
              <FlipHorizontal className="w-4 h-4" />
            </button>
          </div>

          {/* Scanning status */}
          <div className="text-center">
            <div className="flex items-center justify-center gap-2 text-[#6B7280] text-sm">
              {isScanning ? (
                <>
                  <div className="w-2 h-2 bg-[#F97316] rounded-full animate-pulse" />
                  Scanning...
                </>
              ) : (
                <>
                  <div className="w-2 h-2 bg-[#10B981] rounded-full" />
                  Ready to Scan
                </>
              )}
            </div>
          </div>

          {/* Scan Frame button */}
          <button
            onClick={captureFrameForManualScan}
            disabled={isScanning}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-[#2563EB] text-white rounded-lg hover:bg-[#1D4ED8] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isScanning ? (
              <><Loader2 className="w-5 h-5 animate-spin" /> Scanning...</>
            ) : (
              <>📸 Scan Frame</>
            )}
          </button>

          {/* Detection results */}
          {detections.length > 0 && (
            <div className="space-y-2">
              <p className="text-xs text-[#6B7280] font-semibold uppercase tracking-wide">
                Live Detections ({detections.length})
              </p>
              {detections.slice(0, 3).map((det, i) => (
                <div
                  key={i}
                  className="relative p-3 bg-[#FFF7ED] border border-[#F97316]/30 rounded-lg text-sm hover:border-[#F97316] hover:shadow-md transition-all cursor-default"
                  onMouseEnter={() => setHoveredDetectionIndex(i)}
                  onMouseLeave={() => setHoveredDetectionIndex(null)}
                >
                  <div className="flex justify-between items-start">
                    <div>
                      <p className="font-medium text-[#111827]">{det.class_name}</p>
                      <p className="text-[#6B7280] text-xs">{det.category}</p>
                    </div>
                    <span className="text-[#F97316] font-semibold">
                      {(det.classification_confidence * 100).toFixed(1)}%
                    </span>
                  </div>

                  {/* Hover tooltip with top 3 predictions */}
                  {hoveredDetectionIndex === i && det.other_predictions && det.other_predictions.length > 0 && (
                    <div className="absolute top-full left-0 right-0 mt-2 bg-white border border-[#F97316] rounded-lg shadow-lg z-10 p-3 space-y-1">
                      <p className="text-xs font-semibold text-[#111827] mb-2">Top 3 Predictions:</p>
                      {[det, ...(det.other_predictions || [])].slice(0, 3).map((pred, idx) => (
                        <div key={idx} className="flex justify-between text-xs">
                          <span className="text-[#111827]">
                            {idx + 1}. {pred.class_name || det.class_name}
                          </span>
                           <span className="font-semibold text-[#F97316]">
                             {((pred.confidence || det.classification_confidence) * 100).toFixed(1)}%
                           </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <canvas ref={captureCanvasRef} className="hidden" />

      {cameraError && (
        <p className="text-red-500 text-sm mb-4">{cameraError}</p>
      )}
    </div>
  );
}