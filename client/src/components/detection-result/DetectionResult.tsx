'use client';

import { CheckCircle2, AlertTriangle } from 'lucide-react';

interface DetectionResult {
  signName: string;
  confidence: number;
  category: string;
  instruction?: string;
}

interface DetectionResultsProps {
  result: DetectionResult | null;
}

export function DetectionResults({ result }: DetectionResultsProps) {
  if (!result) {
    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h2 className="text-[#111827] mb-4">Detection Results</h2>
        <div className="text-center py-12">
          <AlertTriangle className="w-12 h-12 text-gray-300 mx-auto mb-3" />
          <p className="text-[#6B7280]">No detection results yet</p>
          <p className="text-[#6B7280] mt-1">Upload and detect an image to see results</p>
        </div>
      </div>
    );
  }

  const confidenceColor =
    result.confidence >= 90 ? '#10B981' : result.confidence >= 70 ? '#F97316' : '#6B7280';
  const badgeColor =
    result.confidence >= 90
      ? 'bg-[#10B981]'
      : result.confidence >= 70
        ? 'bg-[#F97316]'
        : 'bg-[#6B7280]';

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <h2 className="text-[#111827] mb-4">Detection Results</h2>

      <div className="space-y-4">
        <div className="flex items-start gap-3">
          <CheckCircle2 className="w-6 h-6 mt-1 flex-shrink-0" style={{ color: confidenceColor }} />
          <div className="flex-1">
            <h3 className="text-[#111827]">{result.signName}</h3>
            <div className="flex items-center gap-2 mt-2">
              <span className={`${badgeColor} text-white px-3 py-1 rounded-full`}>
                {result.confidence}% Confidence
              </span>
              <span className="bg-gray-100 text-[#6B7280] px-3 py-1 rounded-full">
                {result.category}
              </span>
            </div>
          </div>
        </div>

        <div className="border-t border-gray-200 pt-4">
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-[#6B7280]">Detection Status</span>
              <span className="text-[#111827]">Success</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#6B7280]">Category</span>
              <span className="text-[#111827]">{result.category}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#6B7280]">Accuracy</span>
              <span className="text-[#111827]">{result.confidence}%</span>
            </div>
            {result.instruction ? (
              <div className="pt-2 border-t border-gray-100 mt-2">
                <p className="text-[#6B7280] text-sm font-medium mb-1">What to do</p>
                <p className="text-[#111827] text-sm leading-relaxed">{result.instruction}</p>
              </div>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
}
