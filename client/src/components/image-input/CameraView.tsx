import { FlipHorizontal, Loader2, Scan } from 'lucide-react';
import { useRef } from 'react';
import { Detection } from '@/src/lib/api';
import { CameraPermissionError } from './CameraPermissionError';
import { DetectionListItem } from './DetectionListItem';

interface CameraViewProps {
  videoRef: React.RefObject<HTMLVideoElement>;
  canvasRef: React.RefObject<HTMLCanvasElement>;
  cameraError: string | null;
  cameraActive: boolean;
  facingMode: 'environment' | 'user';
  isScanning: boolean;
  detections: Detection[];
  hoveredLiveIndex: number | null;
  onFlipCamera: () => void;
  onCapture: () => void;
  onHoverDetection: (index: number | null) => void;
  onRetryCamera: () => void;
}

export function CameraView({
  videoRef,
  canvasRef,
  cameraError,
  cameraActive,
  facingMode,
  isScanning,
  detections,
  hoveredLiveIndex,
  onFlipCamera,
  onCapture,
  onHoverDetection,
  onRetryCamera,
}: CameraViewProps) {
  return (
    <div className="space-y-4">
      <div className="relative rounded-2xl overflow-hidden bg-[#0F172A] min-h-70 sm:min-h-80 flex items-center justify-center">
        {cameraError === 'permission_denied' ? (
          <CameraPermissionError onRetry={onRetryCamera} />
        ) : (
          <>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full max-h-105 object-contain bg-black"
            />
            <canvas
              ref={canvasRef}
              className="absolute inset-0 w-full h-full pointer-events-none max-h-105"
            />
            {cameraActive && (
              <button
                type="button"
                onClick={onFlipCamera}
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
        onClick={onCapture}
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
          <div className="space-y-2 max-h-55 overflow-y-auto pr-1">
            {[...detections]
              .sort((a, b) => b.classification_confidence - a.classification_confidence)
              .slice(0, 8)
              .map((det, i) => (
                <DetectionListItem
                  key={`${det.class_name}-${det.bbox.join(',')}-${i}`}
                  detection={det}
                  index={i}
                  isHovered={hoveredLiveIndex === i}
                  onHover={onHoverDetection}
                />
              ))}
          </div>
        </div>
      )}
    </div>
  );
}
