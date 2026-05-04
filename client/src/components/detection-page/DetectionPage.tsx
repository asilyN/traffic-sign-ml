'use client';

import { useState, useCallback } from 'react';
import { Navbar } from '@/src/components/navbar';
import { ImageInput } from '@/src/components/image-input';
import { DetectionResults } from '@/src/components/detection-result';
import { DetectionHistory, HistoryItem } from '@/src/components/detection-history';
import { predictImage, Detection } from '@/src/lib/api';

interface DetectionResult {
  predictions: Array<{
    class_id?: number;
    class_name: string;
    confidence: number;
    category?: string;
  }>;
  status: 'success' | 'fail';
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
      // Get top 3 predictions (including the main one)
      const topPredictions = [
        { class_name: prediction.prediction, confidence: prediction.confidence },
        ...(prediction.other_predictions?.slice(0, 2) ?? []),
      ].slice(0, 3);

      const result: DetectionResult = {
        predictions: topPredictions,
        status: 'success',
      };
      setCurrentResult(result);

      const historyItem: HistoryItem = {
        id: Date.now().toString(),
        predictions: topPredictions,
        timestamp: new Date(),
        imageUrl: selectedImage,
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

  // Handle manual frame capture from camera
  const handleFrameCapture = useCallback(
    (imageUrl: string, detections: Detection[]) => {
      if (detections.length === 0) return;

      // Get top 3 predictions
      const topPredictions = detections.slice(0, 3).map((det) => ({
        class_name: det.class_name,
        confidence: det.classification_confidence,
        category: det.category,
      }));

      const result: DetectionResult = {
        predictions: topPredictions,
        status: 'success',
      };
      setCurrentResult(result);

      // Add to history with full detection data (bounding boxes + predictions)
      const historyItem: HistoryItem = {
        id: Date.now().toString(),
        predictions: topPredictions,
        detections: detections, // Save full detection data including bboxes
        timestamp: new Date(),
        imageUrl: imageUrl,
      };
      setHistory((prev) => [historyItem, ...prev]);
    },
    []
  );

  const handleHistoryItemClick = (item: HistoryItem) => {
    const result: DetectionResult = {
      predictions: item.predictions,
      status: 'success',
    };
    setCurrentResult(result);
    setSelectedImage(item.imageUrl);
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
              onFrameCapture={handleFrameCapture}
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
