'use client';

import { useState } from 'react';
import { Navbar } from '@/src/components/navbar';
import { ImageInput } from '@/src/components/image-input';
import { DetectionResults } from '@/src/components/detection-result';
import { DetectionHistory, HistoryItem } from '@/src/components/detection-history';

interface DetectionResult {
  signName: string;
  confidence: number;
  category: string;
}

const mockDetection = (imageUrl: string): DetectionResult => {
  const signs = [
    { signName: 'Stop Sign', category: 'Regulatory', confidence: 95 },
    { signName: 'Yield Sign', category: 'Regulatory', confidence: 88 },
    { signName: 'Speed Limit 50', category: 'Regulatory', confidence: 92 },
    { signName: 'No Entry', category: 'Regulatory', confidence: 97 },
    { signName: 'Pedestrian Crossing', category: 'Warning', confidence: 91 },
    { signName: 'School Zone', category: 'Warning', confidence: 86 },
    { signName: 'Roundabout Ahead', category: 'Warning', confidence: 89 },
    { signName: 'One Way', category: 'Information', confidence: 93 },
  ];

  return signs[Math.floor(Math.random() * signs.length)];
};

export function DetectorPage() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [currentResult, setCurrentResult] = useState<DetectionResult | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);

  const handleImageSelect = (imageUrl: string) => {
    setSelectedImage(imageUrl);
    setCurrentResult(null);
  };

  const handleDetect = async () => {
    if (!selectedImage) return;

    setIsDetecting(true);

    await new Promise((resolve) => setTimeout(resolve, 1500));

    const result = mockDetection(selectedImage);
    setCurrentResult(result);

    const historyItem: HistoryItem = {
      id: Date.now().toString(),
      signName: result.signName,
      confidence: result.confidence,
      timestamp: new Date(),
      imageUrl: selectedImage,
      category: result.category,
    };

    setHistory((prev) => [historyItem, ...prev]);
    setIsDetecting(false);
  };

  const handleHistoryItemClick = (item: HistoryItem) => {
    setSelectedImage(item.imageUrl);
    setCurrentResult({
      signName: item.signName,
      confidence: item.confidence,
      category: item.category,
    });
  };

  return (
    <div className="min-h-screen bg-white flex flex-col">
      <Navbar />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <ImageInput
              selectedImage={selectedImage}
              onImageSelect={handleImageSelect}
              onDetect={handleDetect}
              isDetecting={isDetecting}
            />
          </div>

          <div className="lg:col-span-1 space-y-6">
            <DetectionResults result={currentResult} />
            <DetectionHistory history={history} onItemClick={handleHistoryItemClick} />
          </div>
        </div>
      </main>
    </div>
  );
}
