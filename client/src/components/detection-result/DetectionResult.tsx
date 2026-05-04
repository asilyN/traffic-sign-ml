'use client';

import { CheckCircle2, AlertTriangle } from 'lucide-react';

interface DetectionResult {
  predictions: Array<{
    class_name: string;
    confidence: number;
    category?: string;
  }>;
  status: 'success' | 'fail';
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
          <p className="text-[#6B7280] mt-1">Use camera or upload an image to see results</p>
        </div>
      </div>
    );
  }

  const topPrediction = result.predictions[0];
  const confidenceColor =
    topPrediction.confidence >= 0.9
      ? '#10B981'
      : topPrediction.confidence >= 0.7
        ? '#F97316'
        : '#6B7280';
  const badgeColor =
    topPrediction.confidence >= 0.9
      ? 'bg-[#10B981]'
      : topPrediction.confidence >= 0.7
        ? 'bg-[#F97316]'
        : 'bg-[#6B7280]';

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <h2 className="text-[#111827] mb-4">Detection Results</h2>

      <div className="space-y-4">
        {/* Top prediction */}
        <div className="flex items-start gap-3">
          <CheckCircle2 className="w-6 h-6 mt-1 flex-shrink-0" style={{ color: confidenceColor }} />
          <div className="flex-1">
            <h3 className="text-[#111827] font-semibold">{topPrediction.class_name}</h3>
            <div className="flex items-center gap-2 mt-2">
              <span className={`${badgeColor} text-white px-3 py-1 rounded-full text-sm`}>
                {(topPrediction.confidence * 100).toFixed(1)}%
              </span>
              {topPrediction.category && (
                <span className="bg-gray-100 text-[#6B7280] px-3 py-1 rounded-full text-sm">
                  {topPrediction.category}
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Status section */}
        <div className="border-t border-gray-200 pt-4">
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-[#6B7280]">Detection Status</span>
              <span className="text-[#111827] font-medium">{result.status === 'success' ? 'Success' : 'Failed'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#6B7280]">Confidence</span>
              <span className="text-[#111827] font-medium">{(topPrediction.confidence * 100).toFixed(1)}%</span>
            </div>
            {result.instruction ? (
              <div className="pt-2 border-t border-gray-100 mt-2">
                <p className="text-[#6B7280] text-sm font-medium mb-1">What to do</p>
                <p className="text-[#111827] text-sm leading-relaxed">{result.instruction}</p>
              </div>
            ) : null}
          </div>
        </div>

        {/* All predictions section */}
        {result.predictions.length > 0 && (
          <div className="border-t border-gray-200 pt-4">
            <h4 className="text-[#111827] font-semibold mb-3">Top Predictions</h4>
            <div className="space-y-2">
              {result.predictions.slice(0, 3).map((pred, idx) => (
                <div key={idx} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div className="flex-1 min-w-0">
                    <p className="text-[#111827] font-medium truncate">{idx + 1}. {pred.class_name}</p>
                    {pred.category && (
                      <p className="text-[#6B7280] text-xs">{pred.category}</p>
                    )}
                  </div>
                  <span className="text-[#F97316] font-semibold ml-2 flex-shrink-0">
                    {(pred.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
