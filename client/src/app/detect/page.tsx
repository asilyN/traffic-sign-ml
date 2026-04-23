"use client";
import { useRef, useState, useCallback } from "react";
import Camera from "@/src/components/camera/Camera";
import PredictionBox from "@/src/components/prediction/PredictionBox";
import CaptureButton from "@/src/components/camera/CaptureButton";
import { useCamera } from "@/src/hooks/useCamera";
import { predictImage } from "@/src/lib/api";
import { PredictionResult } from "@/src/types/prediction";

export default function DetectPage() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { videoRef, isStreaming, error: cameraError, startCamera, stopCamera } = useCamera();
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [predError, setPredError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [flash, setFlash] = useState(false);

  const capture = useCallback(async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    setLoading(true);
    setPredError(null);
    setResult(null);
    setFlash(true);
    setTimeout(() => setFlash(false), 200);

    try {
      const ctx = canvas.getContext("2d")!;
      canvas.width = 32;
      canvas.height = 32;
      ctx.drawImage(video, 0, 0, 32, 32);

      const blob = await new Promise<Blob>((resolve, reject) =>
        canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("Capture failed"))), "image/jpeg")
      );

      const res = await predictImage(blob);
      setResult(res);
    } catch (err: any) {
      setPredError(err?.message ?? "Prediction failed. Is the server running?");
    } finally {
      setLoading(false);
    }
  }, [videoRef]);

  const handleToggle = () => {
    if (isStreaming) {
      stopCamera();
      setResult(null);
      setPredError(null);
    } else {
      startCamera();
    }
  };

  return (
    <div className="relative min-h-dvh flex flex-col bg-background">
      {/* Flash overlay */}
      {flash && (
        <div className="fixed inset-0 bg-white/60 z-50 pointer-events-none animate-flash" />
      )}

      {/* Header */}
      <header className="flex items-center justify-between px-4 py-3 border-b border-border bg-card">
        <a
          href="/"
          className="inline-flex items-center gap-1.5 text-muted-foreground hover:text-foreground
                     transition-colors text-xs font-mono"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M19 12H5M12 5l-7 7 7 7"/>
          </svg>
          Back
        </a>

        <div className="flex flex-col items-center">
          <span className="text-[0.6rem] font-mono tracking-widest uppercase text-[var(--color-brand)]">
            AI Detection
          </span>
          <h1 className="text-base font-bold tracking-tight">Traffic Signs</h1>
        </div>

        <div className="flex items-center gap-1.5 text-[0.65rem] font-mono text-muted-foreground tracking-wider">
          <span
            className={`w-2 h-2 rounded-full transition-all ${
              isStreaming
                ? "bg-red-500 shadow-[0_0_6px_theme(colors.red.500)] animate-pulse"
                : "bg-muted-foreground"
            }`}
          />
          {isStreaming ? "LIVE" : "IDLE"}
        </div>
      </header>

      {/* Main */}
      <main className="flex-1 flex flex-col items-center gap-6 p-4 sm:p-6">
        {/* Camera */}
        <div className="relative w-full max-w-2xl">
          <Camera ref={videoRef} isStreaming={isStreaming} />
          <canvas ref={canvasRef} className="hidden" />
          <PredictionBox result={result} error={predError ?? cameraError} />
        </div>

        {/* Controls */}
        <div className="flex items-center gap-6">
          <button
            onClick={handleToggle}
            className={`font-semibold text-sm px-6 py-3 rounded-lg border transition-all ${
              isStreaming
                ? "bg-transparent text-destructive border-destructive/40 hover:bg-destructive/10"
                : "bg-[var(--color-brand)] text-background border-transparent hover:shadow-[0_4px_20px_var(--color-brand-glow)] hover:-translate-y-0.5"
            }`}
          >
            {isStreaming ? "Stop Camera" : "Start Camera"}
          </button>

          {isStreaming && (
            <CaptureButton onClick={capture} loading={loading} disabled={!isStreaming} />
          )}
        </div>

        {result && (
          <p className="text-[0.7rem] font-mono text-muted-foreground tracking-wide">
            ↑ Last detection · tap Capture again to update
          </p>
        )}
      </main>
    </div>
  );
}