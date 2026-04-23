"use client";

import { PredictionResult } from "@/src/types/prediction";

interface PredictionBoxProps {
  result: PredictionResult | null;
  error: string | null;
}

export default function PredictionBox({ result, error }: PredictionBoxProps) {
  if (error) {
    return (
      <div className="absolute bottom-3 left-3 right-3 flex items-center gap-3
                      bg-destructive/10 border border-destructive/40 backdrop-blur-md
                      rounded-lg px-4 py-3 animate-slide-up">
        <span className="text-lg">⚠</span>
        <p className="text-sm text-destructive font-mono">{error}</p>
      </div>
    );
  }

  if (!result) return null;

  const pct = (result.confidence * 100).toFixed(1);
  const barWidth = Math.round(result.confidence * 100);

  return (
    <div className="absolute bottom-3 left-3 right-3
                    bg-background/85 border border-[var(--color-brand)]
                    backdrop-blur-md rounded-lg px-4 py-3
                    shadow-[0_0_20px_var(--color-brand-glow)]
                    animate-slide-up">
      <div className="flex items-center justify-between mb-1">
        <span className="text-[0.65rem] font-mono tracking-widest uppercase text-[var(--color-brand)]">
          Detected Sign
        </span>
        <span className="text-xs font-mono text-[var(--color-brand)]
                         bg-[var(--color-brand-dim)] px-2 py-0.5 rounded">
          {pct}%
        </span>
      </div>

      <p className="text-lg font-bold tracking-tight mb-2">{result.prediction}</p>

      <div className="w-full h-1 bg-muted rounded-full overflow-hidden">
        <div
          className="h-full bg-[var(--color-brand)] rounded-full transition-all duration-500 ease-out"
          style={{ width: `${barWidth}%` }}
        />
      </div>
    </div>
  );
}