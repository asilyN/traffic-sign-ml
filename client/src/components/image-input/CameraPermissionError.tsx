import { CameraOff, RefreshCw } from 'lucide-react';

interface CameraPermissionErrorProps {
  onRetry: () => void;
}

export function CameraPermissionError({ onRetry }: CameraPermissionErrorProps) {
  return (
    <div className="flex flex-col items-center justify-center px-8 py-12 text-center">
      <div className="mb-4 rounded-2xl bg-[#F97316]/15 p-4">
        <CameraOff className="w-14 h-14 text-[#F97316]" strokeWidth={1.5} />
      </div>
      <p className="text-white font-semibold text-lg mb-2">Camera permission denied</p>
      <p className="text-[#94A3B8] text-sm max-w-sm mb-6">
        Allow camera access in your browser settings, then try again to use live detection.
      </p>
      <button
        type="button"
        onClick={onRetry}
        className="inline-flex items-center gap-2 rounded-xl bg-[#F97316] text-white px-5 py-2.5 text-sm font-semibold hover:bg-[#EA580C] transition-colors"
      >
        <RefreshCw className="w-4 h-4" />
        Try Again
      </button>
    </div>
  );
}
