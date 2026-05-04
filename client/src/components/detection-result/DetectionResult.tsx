'use client';

import { CheckCircle2, Info, Scan, Target } from 'lucide-react';
import { useEffect, useState } from 'react';
import { getInstructionForSign } from '@/lib/api';

interface DetectionResult {
  predictions: Array<{
    class_name: string;
    confidence: number;
    category?: string;
    bbox?: [number, number, number, number];
    detection_confidence?: number;
  }>;
  status: 'success' | 'fail';
}

interface DetectionResultsProps {
  result: DetectionResult | null;
}

export function DetectionResults({ result }: DetectionResultsProps) {
  const [topInstruction, setTopInstruction] = useState<string | null>(null);

  useEffect(() => {
    if (!result || result.predictions.length === 0) {
      setTopInstruction(null);
      return;
    }

    const topClassName = result.predictions[0].class_name;
    getInstructionForSign(topClassName)
      .then((instruction) => {
        setTopInstruction(instruction);
      })
      .catch((err) => {
        console.error('Failed to fetch instruction:', err);
        setTopInstruction(null);
      });
  }, [result?.predictions[0]?.class_name]);

  return (
    <div className="rounded-2xl border border-[#E8ECF2] bg-white p-6 shadow-[0_8px_30px_-12px_rgba(15,23,42,0.12)]">
      <div className="flex items-center justify-between gap-3 mb-6">
        <div className="flex items-center gap-2">
          <Target className="w-5 h-5 text-[#F97316]" strokeWidth={2} />
          <h2 className="text-[#0F172A] font-semibold text-[17px] tracking-tight">
            Detection Results
          </h2>
        </div>
        {result && result.predictions.length > 0 && (
          <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 text-emerald-700 px-2.5 py-1 text-xs font-semibold ring-1 ring-emerald-200/80">
            <CheckCircle2 className="w-3.5 h-3.5" />
            Success
          </span>
        )}
      </div>

      {!result || result.predictions.length === 0 ? (
        <div className="text-center py-10 px-4">
          <div className="mx-auto mb-4 inline-flex rounded-2xl bg-[#FFF7ED] p-4 ring-1 ring-[#F97316]/15">
            <Scan className="w-10 h-10 text-[#F97316]" strokeWidth={1.75} />
          </div>
          <p className="text-[#64748B] text-sm leading-relaxed max-w-65 mx-auto">
            Capture or upload a sign, then tap{' '}
            <span className="font-semibold text-[#0F172A]">Capture &amp; Detect</span> or{' '}
            <span className="font-semibold text-[#0F172A]">Detect Sign</span>.
          </p>
        </div>
      ) : (
        <div className="space-y-5">
          {(() => {
            const topPrediction = result.predictions[0];
            const pct = topPrediction.confidence * 100;
            const badgeBg =
              topPrediction.confidence >= 0.9
                ? 'bg-emerald-500'
                : topPrediction.confidence >= 0.7
                  ? 'bg-[#F97316]'
                  : 'bg-[#64748B]';

            return (
              <>
                <div>
                  <div className="flex flex-wrap items-baseline gap-2 mb-3">
                    <h3 className="text-xl font-bold text-[#0F172A] tracking-tight">
                      {topPrediction.class_name}
                    </h3>
                    {topPrediction.category && (
                      <span className="text-xs font-medium text-[#64748B] capitalize px-2 py-1 rounded-full bg-[#F8FAFC]">
                        {topPrediction.category}
                      </span>
                    )}
                  </div>
                </div>

                {topInstruction && (
                  <div className="rounded-xl bg-sky-50 border border-sky-100 px-4 py-3 flex gap-3">
                    <Info className="w-5 h-5 text-sky-600 shrink-0 mt-0.5" />
                    <div>
                      <p className="text-sm font-semibold text-sky-950 mb-1">What to do:</p>
                      <p className="text-sm text-sky-900 leading-snug">{topInstruction}</p>
                    </div>
                  </div>
                )}

                <div className="flex flex-wrap gap-2">
                  <div className="rounded-xl bg-[#F8FAFC] border border-[#E8ECF2] px-3 py-2 text-xs">
                    <span className="text-[#64748B]">Detection Status </span>
                    <span className="font-semibold text-emerald-600">Success</span>
                  </div>
                  <div className="rounded-xl bg-[#F8FAFC] border border-[#E8ECF2] px-3 py-2 text-xs">
                    <span className="text-[#64748B]">Confidence </span>
                    <span className="font-semibold text-[#F97316]">{pct.toFixed(1)}%</span>
                  </div>
                  {topPrediction.detection_confidence !== undefined && (
                    <div className="rounded-xl bg-[#F8FAFC] border border-[#E8ECF2] px-3 py-2 text-xs">
                      <span className="text-[#64748B]">Detection Confidence </span>
                      <span className="font-semibold text-[#F97316]">
                        {(topPrediction.detection_confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  )}
                </div>

                <div className="border-t border-[#EEF2F6] pt-4">
                  <h4 className="text-sm font-semibold text-[#0F172A] mb-3">Top Predictions</h4>
                  <ol className="space-y-2">
                    {result.predictions.slice(0, 3).map((pred, idx) => {
                      const isTop = idx === 0;
                      return (
                        <li
                          key={idx}
                          className="flex items-center justify-between gap-2 rounded-xl bg-[#F8FAFC] border border-[#E8ECF2] px-3 py-2.5"
                        >
                          <span
                            className={`text-sm font-medium truncate ${isTop ? 'text-emerald-600' : 'text-red-500'}`}
                          >
                            {idx + 1}. {pred.class_name}
                          </span>
                          <span
                            className={`text-sm font-bold tabular-nums shrink-0 ${isTop ? 'text-emerald-600' : 'text-red-500'}`}
                          >
                            {(pred.confidence * 100).toFixed(1)}%
                          </span>
                        </li>
                      );
                    })}
                  </ol>
                </div>
              </>
            );
          })()}
        </div>
      )}
    </div>
  );
}
