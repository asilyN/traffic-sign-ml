'use client';

import { useState } from 'react';
import { Navbar } from '@/src/components/navbar';
import { ImageInput } from '@/src/components/image-input';
import { DetectionResults } from '@/src/components/detection-result';
import { DetectionHistory, HistoryItem } from '@/src/components/detection-history';
import { predictImage } from '@/src/lib/api';

interface DetectionResult {
  signName: string;
  confidence: number;
  category: string;
  otherPredictions?: Array<{
    class_id: number;
    class_name: string;
    confidence: number;
  }>;
}

export function DetectorPage() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [currentResult, setCurrentResult] = useState<DetectionResult | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleImageSelect = (imageUrl: string, file: File) => {
    setSelectedImage(imageUrl);
    setSelectedFile(file);
    setCurrentResult(null);
    setErrorMessage(null);
  };

  const handleDetect = async () => {
    if (!selectedImage || !selectedFile) return;

    setIsDetecting(true);
    setErrorMessage(null);

    try {
      const prediction = await predictImage(selectedFile);
      const result: DetectionResult = {
        signName: prediction.prediction,
        confidence: Math.round(prediction.confidence * 100),
        category: 'ML Prediction',
        otherPredictions: prediction.other_predictions?.map((p) => ({
          class_id: p.class_id,
          class_name: p.class_name,
          confidence: Math.round(p.confidence * 100),
        })),
      };
      setCurrentResult(result);

      const historyItem: HistoryItem = {
        id: Date.now().toString(),
        signName: result.signName,
        confidence: result.confidence,
        timestamp: new Date(),
        imageUrl: selectedImage,
        category: result.category,
        otherPredictions: result.otherPredictions,
      };
      setHistory((prev) => [historyItem, ...prev]);
    } catch (error) {
      const fallback = 'Unable to detect sign. Please try another image.';
      const message = error instanceof Error ? error.message : fallback;
      setErrorMessage(message || fallback);
    } finally {
      setIsDetecting(false);
    }
  };

  const handleHistoryItemClick = (item: HistoryItem) => {
    setSelectedImage(item.imageUrl);
    setCurrentResult({
      signName: item.signName,
      confidence: item.confidence,
      category: item.category,
      otherPredictions: item.otherPredictions,
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
            {errorMessage && (
              <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-red-700">
                {errorMessage}
              </div>
            )}
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
