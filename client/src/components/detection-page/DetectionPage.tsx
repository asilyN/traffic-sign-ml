'use client';

import type { CSSProperties } from 'react';
import { useState, useCallback, useRef, useEffect } from 'react';
import Link from 'next/link';
import { ChevronLeft } from 'lucide-react';
import { ImageInput } from '@/src/components/image-input';
import { DetectionResults } from '@/src/components/detection-result';
import { DetectionHistory, HistoryItem } from '@/src/components/detection-history';
import { detectImage, predictImage, type Detection } from '@/src/lib/api';
import { FONT_INTER, FONT_SYNE } from '@/lib/landing-page';

interface DetectionResult {
  predictions: Array<{
    class_id?: number;
    class_name: string;
    confidence: number;
    category?: string;
    bbox?: [number, number, number, number];
    detection_confidence?: number;
  }>;
  status: 'success' | 'fail';
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
  const [uploadPredictions, setUploadPredictions] = useState<Array<{
    class_name: string;
    confidence: number;
  }> | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);
  const isMountedRef = useRef(true);

  // Load history from localStorage on mount
  useEffect(() => {
    if (typeof window === 'undefined') return;

    try {
      const savedHistory = localStorage.getItem('detectionHistory');
      if (savedHistory) {
        const parsed = JSON.parse(savedHistory);
        // Convert ISO strings back to Date objects
        const historyWithDates = parsed.map((item: any) => ({
          ...item,
          timestamp: new Date(item.timestamp),
        }));
        setHistory(historyWithDates);
      }
    } catch (error) {
      console.error('Failed to load history from localStorage:', error);
    }
  }, []);

  // Save history to localStorage whenever it changes
  useEffect(() => {
    if (typeof window === 'undefined') return;

    try {
      // Convert Date objects to ISO strings for serialization
      const historyToSave = history.map((item) => ({
        ...item,
        timestamp: item.timestamp.toISOString(),
      }));
      localStorage.setItem('detectionHistory', JSON.stringify(historyToSave));
    } catch (error) {
      console.error('Failed to save history to localStorage:', error);
    }
  }, [history]);

  useEffect(() => {
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  const handleImageSelect = async (imageUrl: string | null, file: File | null) => {
    setSelectedImage(imageUrl);
    setSelectedFile(file);
    setCurrentResult(null);
    setErrorMessage(null);
    setUploadPredictions(null);

    // Skip automatic detection on upload
    // User must click "Detect Sign" button to run detection
    // This prevents unnecessary YOLO detection which can be slow/inaccurate
  };

  const handleSwitchToCamera = () => {
    setSelectedImage(null);
    setSelectedFile(null);
    setCurrentResult(null);
    setErrorMessage(null);
    setUploadPredictions(null);
  };

   const handleDetect = async () => {
      if (!selectedImage || !selectedFile || selectedFile.size === 0) return;

      // Cancel previous request if any
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }

      // Create new abort controller for this request
      abortControllerRef.current = new AbortController();

      setIsDetecting(true);
      setErrorMessage(null);

      try {
        // Use direct prediction (CNN only) for uploaded images
        const predictionResult = await predictImage(selectedFile);
        
        // Convert prediction to detection format for consistency
        const detection = {
          bbox: [0, 0, 1, 1] as [number, number, number, number],
          class_name: predictionResult.prediction,
          category: predictionResult.category,
          detection_confidence: predictionResult.confidence,
          classification_confidence: predictionResult.confidence,
          other_predictions: predictionResult.other_predictions,
        };

        // Get top 3 predictions including the main prediction
        const topPredictions = [
          {
            class_name: detection.class_name,
            confidence: detection.classification_confidence,
            category: detection.category,
          },
          ...detection.other_predictions.slice(0, 2).map((pred) => ({
            class_name: pred.class_name,
            confidence: pred.confidence,
            category: predictionResult.category,
          })),
        ].slice(0, 3);

        const result: DetectionResult = {
          predictions: topPredictions,
          status: 'success',
        };
        
        if (isMountedRef.current) {
          setCurrentResult(result);
        }

        const historyItem: HistoryItem = {
          id: Date.now().toString(),
          predictions: topPredictions,
          detections: [detection],
          timestamp: new Date(),
          imageUrl: selectedImage,
        };
        
        if (isMountedRef.current) {
          setHistory((prev) => [historyItem, ...prev].slice(0, MAX_HISTORY_ITEMS));
        }
      } catch (error) {
        // Don't show error if request was intentionally aborted
        if (error instanceof Error && error.name === 'AbortError') {
          if (isMountedRef.current) {
            setIsDetecting(false);
          }
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
    };

  // Handle manual frame capture from camera
  const handleFrameCapture = useCallback((imageUrl: string, detections: Detection[]) => {
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
     setHistory((prev) => [historyItem, ...prev].slice(0, MAX_HISTORY_ITEMS));
  }, []);

  const handleHistoryItemClick = (item: HistoryItem) => {
    const result: DetectionResult = {
      predictions: item.predictions,
      status: 'success',
    };
    setCurrentResult(result);
    setSelectedImage(item.imageUrl);
  };

  const handleDeleteHistory = (id: string) => {
    setHistory((prev) => prev.filter((item) => item.id !== id));
  };

  return (
    <div className="min-h-screen bg-[#FAFAFA] flex flex-col" style={{ fontFamily: FONT_INTER }}>
      <header className="bg-white border-b border-[#E5E7EB]">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 pt-4 pb-3">
          <div className="flex items-center justify-between gap-4">
            <Link
              href="/"
              className="inline-flex items-center gap-1 text-[15px] text-[#64748B] hover:text-[#0F172A] transition-colors shrink-0"
            >
              <ChevronLeft className="w-5 h-5" aria-hidden />
              Back
            </Link>
            <div className="font-semibold text-lg tracking-tight" style={{ fontFamily: FONT_SYNE }}>
              <span className="text-[#0F172A]">Traffic</span>
              <span className="text-[#F97316]">Scan</span>
            </div>
            <div className="flex items-center gap-2 text-sm text-[#64748B] shrink-0">
              <span
                className="inline-flex h-2 w-2 rounded-full bg-[#F97316] animate-pulse-dot"
                aria-hidden
              />
              Real-time detection
            </div>
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-6xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <p className="text-[11px] font-semibold tracking-[0.2em] text-[#F97316] uppercase mb-2">
            Detection
          </p>
          <h1 className="text-3xl sm:text-4xl font-bold tracking-tight">
            <span className="text-[#0F172A]">Sign </span>
            <span className="text-[#F97316]">Analyzer</span>
          </h1>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-[1fr_minmax(300px,36%)] gap-8 items-start">
          <div className="space-y-4">
            <ImageInput
              selectedImage={selectedImage}
              onImageSelect={handleImageSelect}
              onSwitchToCamera={handleSwitchToCamera}
              onDetect={handleDetect}
              isDetecting={isDetecting}
              onFrameCapture={handleFrameCapture}
            />
            {errorMessage && (
              <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-red-700 text-sm">
                {errorMessage}
              </div>
            )}
          </div>

          <div className="space-y-6 lg:sticky lg:top-8">
            <DetectionResults result={currentResult} />
            <DetectionHistory
              history={history}
              onItemClick={handleHistoryItemClick}
              onDelete={handleDeleteHistory}
            />
          </div>
        </div>
      </main>
    </div>
  );
}
