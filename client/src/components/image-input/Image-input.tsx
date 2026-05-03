'use client';

import { Upload, Camera, Loader2 } from 'lucide-react';
import { useRef, useCallback, useState } from 'react';
import { compressImage } from '@/src/lib/utils';
import { CameraModal } from '@/src/components/camera-modal';

interface ImageInputProps {
  selectedImage: string | null;
  onImageSelect: (imageUrl: string, file: File) => void;
  onDetect: () => void;
  isDetecting: boolean;
}

export function ImageInput({
  selectedImage,
  onImageSelect,
  onDetect,
  isDetecting,
}: ImageInputProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isCompressing, setIsCompressing] = useState(false);
  const [isCameraOpen, setIsCameraOpen] = useState(false);

  const handleFileChange = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;

      setIsCompressing(true);
      try {
        // Compress image before processing
        const compressedBlob = await compressImage(file);
        const compressedFile = new File([compressedBlob], file.name, {
          type: 'image/jpeg',
        });

        const reader = new FileReader();
        reader.onload = (event) => {
          onImageSelect(event.target?.result as string, compressedFile);
        };
        reader.readAsDataURL(compressedFile);
      } catch (error) {
        console.error('Image compression failed:', error);
        // Fallback: use original file if compression fails
        const reader = new FileReader();
        reader.onload = (event) => {
          onImageSelect(event.target?.result as string, file);
        };
        reader.readAsDataURL(file);
      } finally {
        setIsCompressing(false);
      }
    },
    [onImageSelect]
  );

  const handleCameraCapture = useCallback(
    async (file: File) => {
      setIsCompressing(true);
      try {
        // Compress captured image
        const compressedBlob = await compressImage(file);
        const compressedFile = new File([compressedBlob], file.name, {
          type: 'image/jpeg',
        });

        const reader = new FileReader();
        reader.onload = (event) => {
          onImageSelect(event.target?.result as string, compressedFile);
        };
        reader.readAsDataURL(compressedFile);
      } catch (error) {
        console.error('Image compression failed:', error);
        // Fallback: use original file if compression fails
        const reader = new FileReader();
        reader.onload = (event) => {
          onImageSelect(event.target?.result as string, file);
        };
        reader.readAsDataURL(file);
      } finally {
        setIsCompressing(false);
      }
    },
    [onImageSelect]
  );

  return (
    <>
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h2 className="text-[#111827] mb-4">Image Input</h2>

        <div className="flex flex-col sm:flex-row gap-3 mb-6">
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isCompressing}
            className="flex items-center justify-center gap-2 px-4 py-3 bg-[#F97316] text-white rounded-lg hover:bg-[#EA580C] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Upload className="w-5 h-5" />
            Upload Image
          </button>
          <button
            onClick={() => setIsCameraOpen(true)}
            disabled={isCompressing}
            className="flex items-center justify-center gap-2 px-4 py-3 bg-white text-[#F97316] border-2 border-[#F97316] rounded-lg hover:bg-[#FFF7ED] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Camera className="w-5 h-5" />
            Use Camera
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            disabled={isCompressing}
            className="hidden"
          />
        </div>

        {selectedImage ? (
          <div className="space-y-4">
            <div className="relative border-2 border-gray-200 rounded-lg overflow-hidden bg-gray-50">
              <img
                src={selectedImage}
                alt="Selected"
                className="w-full h-auto max-h-96 object-contain"
              />
            </div>
            <button
              onClick={onDetect}
              disabled={isDetecting || isCompressing}
              className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-[#F97316] text-white rounded-lg hover:bg-[#EA580C] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {isDetecting ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Detecting...
                </>
              ) : isCompressing ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Compressing...
                </>
              ) : (
                'Detect Sign'
              )}
            </button>
          </div>
        ) : (
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-12 text-center">
            <Camera className="w-16 h-16 text-gray-400 mx-auto mb-4" />
            <p className="text-[#6B7280]">No image selected</p>
            <p className="text-[#6B7280] mt-1">Upload an image or use camera to get started</p>
          </div>
        )}
      </div>

      <CameraModal
        isOpen={isCameraOpen}
        onClose={() => setIsCameraOpen(false)}
        onCapture={handleCameraCapture}
      />
    </>
  );
}
