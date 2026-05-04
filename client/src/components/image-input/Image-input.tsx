'use client';

import {
  Upload,
  Loader2,
  FlipHorizontal,
  Camera,
  Scan,
  RefreshCw,
  CameraOff,
} from 'lucide-react';
import { useRef, useState, useCallback, useEffect } from 'react';
import { detectImage, Detection } from '@/src/lib/api';

interface ImageInputProps {
  selectedImage: string | null;
  onImageSelect: (imageUrl: string | null, file: File | null) => void;
  onSwitchToCamera?: () => void;
  onDetect: () => void;
  isDetecting: boolean;
  onFrameCapture?: (imageUrl: string, detections: Detection[]) => void;
}

type TabId = 'camera' | 'upload';

export function ImageInput({
  selectedImage,
  onImageSelect,
  onSwitchToCamera,
  onDetect,
  isDetecting,
  onFrameCapture,
}: ImageInputProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [activeTab, setActiveTab] = useState<TabId>('camera');
  const [cameraActive, setCameraActive] = useState(false);
  const [facingMode, setFacingMode] = useState<'environment' | 'user'>('environment');
  const [isScanning, setIsScanning] = useState(false);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [isDraggingFile, setIsDraggingFile] = useState(false);
  const [hoveredLiveIndex, setHoveredLiveIndex] = useState<number | null>(null);
  const detectionIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isDetectingRef = useRef(false);
  const activeTabRef = useRef<TabId>(activeTab);
  const selectedImageRef = useRef<string | null>(selectedImage);

  activeTabRef.current = activeTab;
  selectedImageRef.current = selectedImage;

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

  const runContinuousDetection = useCallback(async () => {
    const video = videoRef.current;
    const capture = captureCanvasRef.current;
    if (!video || !capture || isDetectingRef.current) return;
    if (video.videoWidth === 0 || video.videoHeight === 0) return;

    try {
      isDetectingRef.current = true;
      capture.width = video.videoWidth;
      capture.height = video.videoHeight;
      capture.getContext('2d')?.drawImage(video, 0, 0);

      capture.toBlob(
        async (blob) => {
          if (!blob) {
            isDetectingRef.current = false;
            return;
          }
          try {
            const data = await detectImage(blob);
            const dets: Detection[] = data.detections ?? [];
            setDetections(dets);
            drawBoxes(dets);
          } catch (err) {
            console.error('Continuous detection failed:', err);
          } finally {
            isDetectingRef.current = false;
          }
        },
        'image/jpeg',
        0.85
      );
    } catch (err) {
      console.error('Frame capture failed:', err);
      isDetectingRef.current = false;
    }
  }, [drawBoxes]);

  useEffect(() => {
    if (activeTab !== 'camera' || !cameraActive || selectedImage) {
      if (detectionIntervalRef.current) {
        clearInterval(detectionIntervalRef.current);
        detectionIntervalRef.current = null;
      }
      return;
    }

    detectionIntervalRef.current = setInterval(() => {
      runContinuousDetection();
    }, 800);

    return () => {
      if (detectionIntervalRef.current) {
        clearInterval(detectionIntervalRef.current);
        detectionIntervalRef.current = null;
      }
    };
  }, [activeTab, cameraActive, selectedImage, runContinuousDetection]);

  const captureFrameForManualScan = useCallback(async () => {
    const video = videoRef.current;
    const capture = captureCanvasRef.current;
    if (!video || !capture || isScanning) return;

    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
      detectionIntervalRef.current = null;
    }

    capture.width = video.videoWidth;
    capture.height = video.videoHeight;
    capture.getContext('2d')?.drawImage(video, 0, 0);

    const resumeContinuousLoop = () => {
      if (
        activeTabRef.current === 'camera' &&
        !selectedImageRef.current &&
        streamRef.current
      ) {
        if (detectionIntervalRef.current) {
          clearInterval(detectionIntervalRef.current);
        }
        detectionIntervalRef.current = setInterval(() => {
          runContinuousDetection();
        }, 800);
      }
    };

    capture.toBlob(
      async (blob) => {
        if (!blob) {
          resumeContinuousLoop();
          return;
        }
        setIsScanning(true);
        try {
          const data = await detectImage(blob);
          const dets: Detection[] = data.detections ?? [];

          setDetections(dets);
          drawBoxes(dets);

          if (dets.length > 0) {
            const topDets = [...dets]
              .sort((a, b) => b.classification_confidence - a.classification_confidence)
              .slice(0, 3);

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
          resumeContinuousLoop();
        }
      },
      'image/jpeg',
      0.92
    );
  }, [isScanning, onFrameCapture, drawBoxes, runContinuousDetection]);

  const stopCamera = useCallback(() => {
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
      detectionIntervalRef.current = null;
    }
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setCameraActive(false);
    setDetections([]);
    const cv = canvasRef.current?.getContext('2d');
    if (canvasRef.current && cv) {
      cv.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  }, []);

  const startCamera = useCallback(
    async (facing: 'environment' | 'user' = facingMode) => {
      setCameraError(null);
      streamRef.current?.getTracks().forEach((t) => t.stop());
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: facing,
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
        });
        streamRef.current = stream;
        if (videoRef.current) videoRef.current.srcObject = stream;
        setCameraActive(true);
      } catch (err) {
        setCameraError('permission_denied');
        console.error(err);
      }
    },
    [facingMode]
  );

  useEffect(() => {
    if (!selectedImage) return;
    setActiveTab('upload');
    stopCamera();
  }, [selectedImage, stopCamera]);

  useEffect(() => {
    if (activeTab !== 'camera' || selectedImage) {
      stopCamera();
      return;
    }
    startCamera();
    return () => {
      stopCamera();
    };
  }, [activeTab, selectedImage, startCamera, stopCamera]);

  const selectTab = (tab: TabId) => {
    if (tab === 'camera') {
      setActiveTab('camera');
      onSwitchToCamera?.();
    } else {
      setActiveTab('upload');
      stopCamera();
    }
  };

  const applyFile = (file: File | undefined) => {
    if (!file || !file.type.startsWith('image/')) return;
    stopCamera();
    setActiveTab('upload');
    const reader = new FileReader();
    reader.onload = (event) =>
      onImageSelect(event.target?.result as string, file);
    reader.readAsDataURL(file);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    applyFile(file);
    e.target.value = '';
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDraggingFile(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDraggingFile(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDraggingFile(false);
    const file = e.dataTransfer.files?.[0];
    applyFile(file);
  };

  const handleClearImage = () => {
    onImageSelect(null, null);
    setDetections([]);
    if (activeTab === 'camera') {
      startCamera();
    }
  };

  const tabBtn =
    'relative flex flex-1 items-center justify-center gap-2 rounded-xl px-4 py-3 text-sm font-semibold transition-all duration-200';

  return (
    <div className="space-y-5">
      {/* Segmented tabs */}
      <div className="rounded-2xl bg-[#EEF2F6] p-1.5 flex gap-1 shadow-inner">
        <button
          type="button"
          onClick={() => selectTab('camera')}
          className={`${tabBtn} ${
            activeTab === 'camera'
              ? 'bg-white text-[#F97316] shadow-md shadow-black/5'
              : 'text-[#64748B] hover:text-[#0F172A]'
          }`}
        >
          <Camera className="w-4 h-4 shrink-0" />
          Camera
        </button>
        <button
          type="button"
          onClick={() => selectTab('upload')}
          className={`${tabBtn} ${
            activeTab === 'upload'
              ? 'bg-white text-[#F97316] shadow-md shadow-black/5'
              : 'text-[#64748B] hover:text-[#0F172A]'
          }`}
        >
          <Upload className="w-4 h-4 shrink-0" />
          Upload Image
        </button>
      </div>

      {/* Camera */}
      {activeTab === 'camera' && (
        <div className="space-y-4">
          <div className="relative rounded-2xl overflow-hidden bg-[#0F172A] min-h-[280px] sm:min-h-[320px] flex items-center justify-center">
            {cameraError === 'permission_denied' ? (
              <div className="flex flex-col items-center justify-center px-8 py-12 text-center">
                <div className="mb-4 rounded-2xl bg-[#F97316]/15 p-4">
                  <CameraOff className="w-14 h-14 text-[#F97316]" strokeWidth={1.5} />
                </div>
                <p className="text-white font-semibold text-lg mb-2">
                  Camera permission denied
                </p>
                <p className="text-[#94A3B8] text-sm max-w-sm mb-6">
                  Allow camera access in your browser settings, then try again to use live
                  detection.
                </p>
                <button
                  type="button"
                  onClick={() => startCamera()}
                  className="inline-flex items-center gap-2 rounded-xl bg-[#F97316] text-white px-5 py-2.5 text-sm font-semibold hover:bg-[#EA580C] transition-colors"
                >
                  <RefreshCw className="w-4 h-4" />
                  Try Again
                </button>
              </div>
            ) : (
              <>
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full max-h-[420px] object-contain bg-black"
                />
                <canvas
                  ref={canvasRef}
                  className="absolute inset-0 w-full h-full pointer-events-none max-h-[420px]"
                />
                {cameraActive && (
                  <button
                    type="button"
                    onClick={() => {
                      const next =
                        facingMode === 'environment' ? 'user' : 'environment';
                      setFacingMode(next);
                      startCamera(next);
                    }}
                    className="absolute top-3 left-3 p-2 rounded-full bg-black/50 text-white hover:bg-black/70 transition-colors"
                    aria-label="Flip camera"
                  >
                    <FlipHorizontal className="w-4 h-4" />
                  </button>
                )}
              </>
            )}
          </div>

          <button
            type="button"
            onClick={() => captureFrameForManualScan()}
            disabled={isScanning || !!cameraError || !cameraActive}
            className="w-full flex items-center justify-center gap-2 rounded-xl bg-[#F97316] text-white px-4 py-4 text-[15px] font-semibold hover:bg-[#EA580C] disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-lg shadow-orange-500/25"
          >
            {isScanning ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Detecting…
              </>
            ) : (
              <>
                <Scan className="w-5 h-5" />
                Capture &amp; Detect
              </>
            )}
          </button>

          {cameraActive && !cameraError && detections.length > 0 && (
            <div className="space-y-2 pt-1">
              <p className="text-[11px] font-semibold uppercase tracking-wider text-[#64748B]">
                Live detection ({detections.length})
              </p>
              <div className="space-y-2 max-h-[220px] overflow-y-auto pr-1">
                {[...detections]
                  .sort(
                    (a, b) =>
                      b.classification_confidence - a.classification_confidence
                  )
                  .slice(0, 8)
                  .map((det, i) => (
                    <div
                      key={`${det.class_name}-${det.bbox.join(',')}-${i}`}
                      className="relative rounded-xl border border-[#F97316]/25 bg-[#FFF7ED] p-3 text-sm hover:border-[#F97316]/50 hover:shadow-sm transition-all"
                      onMouseEnter={() => setHoveredLiveIndex(i)}
                      onMouseLeave={() => setHoveredLiveIndex(null)}
                    >
                      <div className="flex justify-between items-start gap-2">
                        <div className="min-w-0">
                          <p className="font-semibold text-[#0F172A] truncate">
                            {det.class_name}
                          </p>
                          <p className="text-[#64748B] text-xs">{det.category}</p>
                        </div>
                        <span className="text-[#F97316] font-bold tabular-nums shrink-0">
                          {(det.classification_confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      {hoveredLiveIndex === i &&
                        det.other_predictions &&
                        det.other_predictions.length > 0 && (
                          <div className="absolute left-0 right-0 top-full z-10 mt-2 rounded-xl border border-[#F97316]/30 bg-white p-3 shadow-lg space-y-1.5">
                            <p className="text-[11px] font-semibold text-[#0F172A] uppercase tracking-wide">
                              Top predictions
                            </p>
                            {[det, ...det.other_predictions]
                              .slice(0, 3)
                              .map((pred, idx) => {
                                const label =
                                  idx === 0
                                    ? det.class_name
                                    : pred.class_name;
                                const conf =
                                  idx === 0
                                    ? det.classification_confidence
                                    : pred.confidence;
                                return (
                                <div
                                  key={idx}
                                  className="flex justify-between gap-2 text-xs"
                                >
                                  <span className="text-[#334155] truncate">
                                    {idx + 1}. {label}
                                  </span>
                                  <span className="font-semibold text-[#F97316] shrink-0">
                                    {(conf * 100).toFixed(1)}%
                                  </span>
                                </div>
                              );
                              })}
                          </div>
                        )}
                    </div>
                  ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Upload */}
      {activeTab === 'upload' && (
        <div className="space-y-4">
          {!selectedImage ? (
            <>
              <button
                type="button"
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`relative w-full rounded-2xl border-2 border-dashed transition-colors min-h-[280px] flex flex-col items-center justify-center px-6 py-10 cursor-pointer bg-[#F8FAFC] hover:bg-[#F1F5F9] ${
                  isDraggingFile
                    ? 'border-[#F97316] bg-orange-50/50'
                    : 'border-[#CBD5E1]'
                }`}
              >
                <div className="mb-4 rounded-2xl bg-[#FFF7ED] p-4 ring-1 ring-[#F97316]/20">
                  <Upload className="w-10 h-10 text-[#F97316]" strokeWidth={2} />
                </div>
                <p className="text-[#0F172A] font-medium text-center mb-1">
                  Drop an image here or click the button below to browse.
                </p>
                <p className="text-[#64748B] text-xs mt-4 flex items-center gap-3">
                  <span className="h-px w-10 bg-[#E2E8F0]" aria-hidden />
                  PNG · JPG · WEBP
                  <span className="h-px w-10 bg-[#E2E8F0]" aria-hidden />
                </p>
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/png,image/jpeg,image/webp,image/*"
                onChange={handleFileChange}
                className="hidden"
              />
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="w-full flex items-center justify-center gap-2 rounded-xl border-2 border-[#E2E8F0] bg-white text-[#0F172A] px-4 py-3.5 text-sm font-semibold hover:border-[#CBD5E1] hover:bg-[#FAFAFA] transition-colors"
              >
                <Upload className="w-5 h-5 text-[#64748B]" />
                Browse Files
              </button>
              <button
                type="button"
                disabled
                className="w-full flex items-center justify-center gap-2 rounded-xl bg-[#F97316]/40 text-white px-4 py-4 text-[15px] font-semibold cursor-not-allowed"
              >
                <Scan className="w-5 h-5" />
                Detect Sign
              </button>
            </>
          ) : (
            <>
              <div className="relative rounded-2xl border-2 border-[#E2E8F0] overflow-hidden bg-[#F8FAFC]">
                <button
                  type="button"
                  onClick={handleClearImage}
                  className="absolute top-3 right-3 z-10 inline-flex items-center gap-1.5 rounded-full bg-white/95 border border-[#E2E8F0] px-3 py-1.5 text-xs font-semibold text-[#64748B] shadow-sm hover:bg-white hover:text-[#0F172A] transition-colors"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                  Reset
                </button>
                <img
                  src={selectedImage}
                  alt="Selected sign"
                  className="w-full h-auto max-h-[380px] object-contain mx-auto block"
                />
              </div>
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="w-full flex items-center justify-center gap-2 rounded-xl border-2 border-[#E2E8F0] bg-white text-[#0F172A] px-4 py-3.5 text-sm font-semibold hover:border-[#CBD5E1] hover:bg-[#FAFAFA] transition-colors"
              >
                <Upload className="w-5 h-5 text-[#64748B]" />
                Browse Files
              </button>
              <button
                type="button"
                onClick={onDetect}
                disabled={isDetecting}
                className="w-full flex items-center justify-center gap-2 rounded-xl bg-[#F97316] text-white px-4 py-4 text-[15px] font-semibold hover:bg-[#EA580C] disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-lg shadow-orange-500/25"
              >
                {isDetecting ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Detecting…
                  </>
                ) : (
                  <>
                    <Scan className="w-5 h-5" />
                    Detect Sign
                  </>
                )}
              </button>
            </>
          )}
        </div>
      )}

      <canvas ref={captureCanvasRef} className="hidden" />
    </div>
  );
}