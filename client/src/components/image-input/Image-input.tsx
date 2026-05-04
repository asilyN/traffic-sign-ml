'use client';

import { useRef, useState, useCallback, useEffect } from 'react';
import { detectImage, Detection } from '@/src/lib/api';
import { TabSelector } from './TabSelector';
import { CameraView } from './CameraView';
import { UploadView } from './UploadView';
import { drawBoxes, clearCanvas } from './utils/canvasUtils';
import { performDetection, captureFrame } from './utils/detectionUtils';
import { startCameraStream, stopCameraStream } from './utils/cameraUtils';
import { isValidImageFile, fileToDataUrl } from './utils/fileUtils';

interface ImageInputProps {
  selectedImage: string | null;
  onImageSelect: (imageUrl: string | null, file: File | null) => void;
  onSwitchToCamera?: () => void;
  onDetect: () => void;
  isDetecting: boolean;
  onFrameCapture?: (imageUrl: string, detections: Detection[]) => void;
}

type TabId = 'camera' | 'upload';

export function ImageInput({
  selectedImage,
  onImageSelect,
  onSwitchToCamera,
  onDetect,
  isDetecting,
  onFrameCapture,
}: ImageInputProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [activeTab, setActiveTab] = useState<TabId>('camera');
  const [cameraActive, setCameraActive] = useState(false);
  const [facingMode, setFacingMode] = useState<'environment' | 'user'>('environment');
  const [isScanning, setIsScanning] = useState(false);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [isDraggingFile, setIsDraggingFile] = useState(false);
  const [hoveredLiveIndex, setHoveredLiveIndex] = useState<number | null>(null);
  const detectionIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isDetectingRef = useRef(false);
  const activeTabRef = useRef<TabId>(activeTab);
  const selectedImageRef = useRef<string | null>(selectedImage);

  activeTabRef.current = activeTab;
  selectedImageRef.current = selectedImage;

  const stopCamera = useCallback(() => {
    if (detectionIntervalRef.current) clearInterval(detectionIntervalRef.current);
    stopCameraStream(streamRef.current);
    streamRef.current = null;
    setCameraActive(false);
    setDetections([]);
    if (canvasRef.current) clearCanvas(canvasRef.current);
  }, []);

  const runContinuousDetection = useCallback(async () => {
    const video = videoRef.current;
    const capture = captureCanvasRef.current;
    const canvas = canvasRef.current;
    if (!video || !capture || !canvas || isDetectingRef.current) return;

    try {
      isDetectingRef.current = true;
      const dets = await performDetection(video, capture, canvas);
      setDetections(dets);
    } finally {
      isDetectingRef.current = false;
    }
  }, []);

  useEffect(() => {
    if (activeTab !== 'camera' || !cameraActive || selectedImage) {
      if (detectionIntervalRef.current) clearInterval(detectionIntervalRef.current);
      return;
    }
    detectionIntervalRef.current = setInterval(runContinuousDetection, 800);
    return () => {
      if (detectionIntervalRef.current) clearInterval(detectionIntervalRef.current);
    };
  }, [activeTab, cameraActive, selectedImage, runContinuousDetection]);

  const captureFrameForManualScan = useCallback(async () => {
    const video = videoRef.current;
    const capture = captureCanvasRef.current;
    const canvas = canvasRef.current;
    if (!video || !capture || !canvas || isScanning) return;

    if (detectionIntervalRef.current) clearInterval(detectionIntervalRef.current);

    const resumeLoop = () => {
      if (activeTabRef.current === 'camera' && !selectedImageRef.current && streamRef.current) {
        detectionIntervalRef.current = setInterval(runContinuousDetection, 800);
      }
    };

    setIsScanning(true);
    try {
      const dets = await performDetection(video, capture, canvas);
      if (dets.length > 0) {
        const topDets = dets
          .sort((a, b) => b.classification_confidence - a.classification_confidence)
          .slice(0, 3);
        const frameData = await captureFrame(video, capture, 0.92);
        if (frameData?.blob) {
          const reader = new FileReader();
          reader.onload = (e) => {
            const imageUrl = e.target?.result as string;
            onFrameCapture?.(imageUrl, topDets);
          };
          reader.readAsDataURL(frameData.blob);
        }
      }
    } finally {
      setIsScanning(false);
      resumeLoop();
    }
  }, [isScanning, onFrameCapture, runContinuousDetection]);

  const startCamera = useCallback(
    async (facing: 'environment' | 'user' = facingMode) => {
      setCameraError(null);
      stopCameraStream(streamRef.current);
      try {
        const stream = await startCameraStream(videoRef.current!, facing);
        if (stream) {
          streamRef.current = stream;
          setCameraActive(true);
        } else {
          setCameraError('permission_denied');
        }
      } catch (err) {
        setCameraError('permission_denied');
        console.error(err);
      }
    },
    [facingMode]
  );

  useEffect(() => {
    if (!selectedImage) return;
    setActiveTab('upload');
    stopCamera();
  }, [selectedImage, stopCamera]);

  useEffect(() => {
    if (activeTab !== 'camera' || selectedImage) {
      stopCamera();
      return;
    }
    startCamera();
    return stopCamera;
  }, [activeTab, selectedImage, startCamera, stopCamera]);

  const selectTab = (tab: TabId) => {
    setActiveTab(tab);
    if (tab === 'camera') onSwitchToCamera?.();
    else stopCamera();
  };

  const applyFile = useCallback(
    (file: File | undefined) => {
      if (!isValidImageFile(file)) return;
      stopCamera();
      setActiveTab('upload');
      fileToDataUrl(file!).then((dataUrl) => onImageSelect(dataUrl, file!));
    },
    [stopCamera, onImageSelect]
  );

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    applyFile(file);
    e.target.value = '';
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDraggingFile(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDraggingFile(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDraggingFile(false);
    applyFile(e.dataTransfer.files?.[0]);
  };

  const handleClearImage = () => {
    onImageSelect(null, null);
    setDetections([]);
    if (activeTab === 'camera') startCamera();
  };

  const handleFlipCamera = () => {
    const next = facingMode === 'environment' ? 'user' : 'environment';
    setFacingMode(next);
    startCamera(next);
  };

  return (
    <div className="space-y-5">
      <TabSelector activeTab={activeTab} onTabChange={selectTab} />

      {activeTab === 'camera' && (
        <CameraView
          videoRef={videoRef as React.RefObject<HTMLVideoElement>}
          canvasRef={canvasRef as React.RefObject<HTMLCanvasElement>}
          cameraError={cameraError}
          cameraActive={cameraActive}
          facingMode={facingMode}
          isScanning={isScanning}
          detections={detections}
          hoveredLiveIndex={hoveredLiveIndex}
          onFlipCamera={handleFlipCamera}
          onCapture={captureFrameForManualScan}
          onHoverDetection={setHoveredLiveIndex}
          onRetryCamera={() => startCamera()}
        />
      )}

      {activeTab === 'upload' && (
        <UploadView
          selectedImage={selectedImage}
          isDetecting={isDetecting}
          isDraggingFile={isDraggingFile}
          fileInputRef={fileInputRef as React.RefObject<HTMLInputElement>}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onFileClick={() => fileInputRef.current?.click()}
          onFileChange={handleFileChange}
          onDetect={onDetect}
          onClear={handleClearImage}
        />
      )}

      <canvas ref={captureCanvasRef} className="hidden" />
    </div>
  );
}
