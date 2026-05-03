'use client';

import { Upload, Camera, Loader2, X, FlipHorizontal } from 'lucide-react';
import { useRef, useState, useCallback, useEffect } from 'react';
import { detectImage, Detection } from '@/src/lib/api';
interface ImageInputProps {
  selectedImage: string | null;
  onImageSelect: (imageUrl: string, file: File) => void;
  onDetect: () => void;
  isDetecting: boolean;
  onDetections?: (detections: Detection[]) => void;
}

export function ImageInput({
  selectedImage,
  onImageSelect,
  onDetect,
  isDetecting,
  onDetections,
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

  // ── Capture frame → detectImage() → draw YOLO results ────────────────────
  const runDetection = useCallback(async () => {
    const video = videoRef.current;
    const capture = captureCanvasRef.current;
    if (!video || !capture || isScanning) return;

    capture.width = video.videoWidth;
    capture.height = video.videoHeight;
    capture.getContext('2d')?.drawImage(video, 0, 0);

    capture.toBlob(async (blob) => {
      if (!blob) return;
      setIsScanning(true);
      try {
        const data = await detectImage(blob);
        const dets: Detection[] = data.detections ?? [];
        setDetections(dets);
        drawBoxes(dets);
        onDetections?.(dets);
      } catch (err) {
        console.error('Detection failed:', err);
      } finally {
        setIsScanning(false);
      }
    }, 'image/jpeg', 0.92);
  }, [isScanning, drawBoxes, onDetections]);

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
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setCameraActive(false);
    setDetections([]);
  }, []);

  useEffect(() => () => stopCamera(), [stopCamera]);

  const flipCamera = () => {
    const next = facingMode === 'environment' ? 'user' : 'environment';
    setFacingMode(next);
    startCamera(next);
  };

  // ── Upload handler ─────────────────────────────────────────────────────────
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => onImageSelect(event.target?.result as string, file);
    reader.readAsDataURL(file);
  };

  return (
    <>
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h2 className="text-[#111827] mb-4">Image Input</h2>

      {!cameraActive && (
        <div className="flex flex-col sm:flex-row gap-3 mb-6">
          <button
            onClick={() => fileInputRef.current?.click()}
            className="flex items-center justify-center gap-2 px-4 py-3 bg-[#F97316] text-white rounded-lg hover:bg-[#EA580C] transition-colors"
          >
            <Upload className="w-5 h-5" />
            Upload Image
          </button>
          <button
            onClick={() => startCamera()}
            className="flex items-center justify-center gap-2 px-4 py-3 bg-white text-[#F97316] border-2 border-[#F97316] rounded-lg hover:bg-[#FFF7ED] transition-colors"
          >
            <Camera className="w-5 h-5" />
            Use Camera
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="hidden"
          />
        </div>
      )}

      {cameraActive && (
        <div className="space-y-3 mb-4">
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
              onClick={stopCamera}
              className="absolute top-2 right-2 p-1.5 bg-black/60 text-white rounded-full hover:bg-black/80 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
            <button
              onClick={flipCamera}
              className="absolute top-2 left-2 p-1.5 bg-black/60 text-white rounded-full hover:bg-black/80 transition-colors"
            >
              <FlipHorizontal className="w-4 h-4" />
            </button>
          </div>

          <button
            onClick={runDetection}
            disabled={isScanning}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-[#F97316] text-white rounded-lg hover:bg-[#EA580C] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isScanning ? (
              <><Loader2 className="w-5 h-5 animate-spin" /> Detecting...</>
            ) : (
              <><Camera className="w-5 h-5" /> Scan Frame</>
            )}
          </button>

          {detections.length > 0 && (
            <div className="space-y-2">
              {detections.map((det, i) => (
                <div
                  key={i}
                  className="flex justify-between items-center p-3 bg-[#FFF7ED] border border-[#F97316]/30 rounded-lg text-sm"
                >
                  <div>
                    <p className="font-medium text-[#111827]">{det.class_name}</p>
                    <p className="text-[#6B7280]">{det.category}</p>
                  </div>
                  <span className="text-[#F97316] font-semibold">
                    {(det.classification_confidence * 100).toFixed(1)}%
                  </span>
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

      {!cameraActive && selectedImage ? (
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
        </div>
      ) : !cameraActive && (
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-12 text-center">
          <Camera className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <p className="text-[#6B7280]">No image selected</p>
          <p className="text-[#6B7280] mt-1">Upload an image or use camera to get started</p>
        </div>
      )}
    </div>
  );
}