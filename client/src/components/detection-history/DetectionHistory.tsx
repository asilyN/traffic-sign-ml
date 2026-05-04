'use client';

import { Clock } from 'lucide-react';

export interface HistoryItem {
  id: string;
  signName: string;
  confidence: number;
  timestamp: Date;
  imageUrl: string;
  category: string;
  otherPredictions?: Array<{
    class_id: number;
    class_name: string;
    confidence: number;
  }>;
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
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <h2 className="text-[#111827] mb-4">Detection History</h2>

      {history.length === 0 ? (
        <div className="text-center py-12">
          <Clock className="w-12 h-12 text-gray-300 mx-auto mb-3" />
          <p className="text-[#6B7280]">No history yet</p>
          <p className="text-[#6B7280] mt-1">Your detections will appear here</p>
        </div>
      ) : (
        <div className="space-y-3 max-h-96 overflow-y-auto">
          {history.map((item) => (
            <button
              key={item.id}
              onClick={() => onItemClick(item)}
              className="w-full flex items-center gap-3 p-3 rounded-lg border border-gray-200 hover:border-[#2563EB] hover:bg-[#eff6ff] transition-all group"
            >
              <div className="w-16 h-16 flex-shrink-0 rounded-lg overflow-hidden bg-gray-100">
                <img
                  src={item.imageUrl}
                  alt={item.signName}
                  className="w-full h-full object-cover"
                />
              </div>
              <div className="flex-1 text-left">
                <h4 className="text-[#111827] group-hover:text-[#2563EB] transition-colors">
                  {item.signName}
                </h4>
                <div className="flex items-center gap-2 mt-1">
                  <span className="text-[#6B7280]">{item.confidence}%</span>
                  <span className="text-[#6B7280]">•</span>
                  <span className="text-[#6B7280]">{formatTime(item.timestamp)}</span>
                </div>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
