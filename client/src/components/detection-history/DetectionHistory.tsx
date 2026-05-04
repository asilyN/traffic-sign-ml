'use client';

import { Clock } from 'lucide-react';

export interface HistoryItem {
  id: string;
  predictions: Array<{
    class_name: string;
    confidence: number;
    category?: string;
  }>;
  detections?: Array<{
    bbox: [number, number, number, number];
    class_name: string;
    classification_confidence: number;
    other_predictions?: Array<{
      class_name: string;
      confidence: number;
    }>;
  }>;
  timestamp: Date;
  imageUrl: string;
}

interface DetectionHistoryProps {
  history: HistoryItem[];
  onItemClick: (item: HistoryItem) => void;
}

export function DetectionHistory({ history, onItemClick }: DetectionHistoryProps) {
  const formatTime = (date: Date) => {
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const minutes = Math.floor(diff / 60000);

    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    return date.toLocaleDateString();
  };

  return (
    <div className="rounded-2xl border border-[#E8ECF2] bg-white p-6 shadow-[0_8px_30px_-12px_rgba(15,23,42,0.12)]">
      <div className="flex items-center gap-2 mb-5">
        <Clock className="w-5 h-5 text-[#F97316]" strokeWidth={2} />
        <h2 className="text-[#0F172A] font-semibold text-[17px] tracking-tight">
          Detection History
        </h2>
      </div>

      {history.length === 0 ? (
        <div className="text-center py-10 px-4">
          <Clock
            className="w-11 h-11 text-[#CBD5E1] mx-auto mb-3"
            strokeWidth={1.25}
          />
          <p className="text-[#64748B] text-sm">No detections yet.</p>
        </div>
      ) : (
        <div className="space-y-2 max-h-[min(420px,55vh)] overflow-y-auto pr-1">
          {history.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => onItemClick(item)}
              className="w-full flex items-center gap-3 p-3 rounded-xl border border-[#EEF2F6] bg-[#FAFBFC] hover:border-[#F97316]/40 hover:bg-white hover:shadow-sm transition-all text-left group"
            >
              <div className="w-14 h-14 shrink-0 rounded-xl overflow-hidden bg-[#EEF2F6] ring-1 ring-black/4">
                <img
                  src={item.imageUrl}
                  alt={item.predictions[0]?.class_name || 'Detection'}
                  className="w-full h-full object-cover"
                />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-[#0F172A] font-semibold text-sm truncate group-hover:text-[#F97316] transition-colors">
                  {item.predictions[0]?.class_name || 'Unknown'}
                </p>
                <div className="flex items-center gap-2 mt-1 text-xs text-[#64748B]">
                  <span className="font-medium text-[#F97316]">
                    {((item.predictions[0]?.confidence ?? 0) * 100).toFixed(0)}%
                  </span>
                  <span aria-hidden>·</span>
                  <span>{formatTime(item.timestamp)}</span>
                </div>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
