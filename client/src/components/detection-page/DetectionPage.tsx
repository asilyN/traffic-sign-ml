'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import { Navbar } from '@/src/components/navbar';
import { ImageInput } from '@/src/components/image-input';
import { DetectionResults } from '@/src/components/detection-result';
import { DetectionHistory, HistoryItem } from '@/src/components/detection-history';
import { predictImage } from '@/src/lib/api';
import { generateThumbnail, preprocessImageForCNN } from '@/src/lib/utils';

interface DetectionResult {
  signName: string;
  confidence: number;
  category: string;
}

// Max history items to keep in memory (prevent unbounded growth)
const MAX_HISTORY_ITEMS = 10;

export function DetectorPage() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [currentResult, setCurrentResult] = useState<DetectionResult | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // For cancelling in-flight requests
  const abortControllerRef = useRef<AbortController | null>(null);
  const isMountedRef = useRef(true);

  useEffect(() => {
    return () => {
      isMountedRef.current = false;
      // Cancel any pending requests on unmount
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  const handleImageSelect = useCallback((imageUrl: string, file: File) => {
    setSelectedImage(imageUrl);
    setSelectedFile(file);
    setCurrentResult(null);
    setErrorMessage(null);
  }, []);

  const handleDetect = useCallback(async () => {
    if (!selectedImage || !selectedFile) return;

    // Cancel previous request if any
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    // Create new abort controller for this request
    abortControllerRef.current = new AbortController();

    setIsDetecting(true);
    setErrorMessage(null);

    try {
      // Preprocess image for visualization and validation
      try {
        const preprocessed = await preprocessImageForCNN(selectedFile);
        console.log('✓ Image preprocessing complete:', {
          size: `${preprocessed.width}x${preprocessed.height}`,
          normalizedSample: preprocessed.normalized.slice(0, 9), // First 3 pixels worth of data
        });
      } catch (preprocessErr) {
        console.warn(
          'Preprocessing visualization failed (will not affect prediction):',
          preprocessErr
        );
      }

      const prediction = await predictImage(selectedFile, abortControllerRef.current.signal);

      // Check if component is still mounted before updating state
      if (!isMountedRef.current) return;

      const result: DetectionResult = {
        signName: prediction.prediction,
        confidence: Math.round(prediction.confidence * 100),
        category: prediction.category ?? 'unknown',
      };
      setCurrentResult(result);

      const historyItem: HistoryItem = {
        id: Date.now().toString(),
        signName: result.signName,
        confidence: result.confidence,
        timestamp: new Date(),
        category: result.category,
        thumbnail: undefined, // Will be set asynchronously
      };

      // Add to history first (will show without thumbnail initially)
      setHistory((prev) => [historyItem, ...prev].slice(0, MAX_HISTORY_ITEMS));

      // Generate thumbnail asynchronously to avoid blocking
      generateThumbnail(selectedFile)
        .then((thumb) => {
          setHistory((prev) =>
            prev.map((item) => (item.id === historyItem.id ? { ...item, thumbnail: thumb } : item))
          );
        })
        .catch((err) => {
          console.warn('Failed to generate thumbnail:', err);
          // History item will still show without thumbnail (badge fallback)
        });
    } catch (error) {
      // Don't show error if request was intentionally aborted
      if (error instanceof Error && error.name === 'AbortError') {
        return;
      }

      if (!isMountedRef.current) return;

      const fallback = 'Unable to detect sign. Please try another image.';
      const message = error instanceof Error ? error.message : fallback;
      setErrorMessage(message || fallback);
    } finally {
      if (isMountedRef.current) {
        setIsDetecting(false);
      }
    }
  }, [selectedImage, selectedFile]);

  const handleHistoryItemClick = useCallback((item: HistoryItem) => {
    // Keep current selectedImage and file - just show the result
    setCurrentResult({
      signName: item.signName,
      confidence: item.confidence,
      category: item.category,
    });
  }, []);

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
