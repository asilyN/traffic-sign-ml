'use client';

import { Upload, Camera, Loader2 } from 'lucide-react';
import { useRef } from 'react';

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
  const cameraInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        onImageSelect(event.target?.result as string, file);
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <h2 className="text-[#111827] mb-4">Image Input</h2>

      <div className="flex flex-col sm:flex-row gap-3 mb-6">
        <button
          onClick={() => fileInputRef.current?.click()}
          className="flex items-center justify-center gap-2 px-4 py-3 bg-[#F97316] text-white rounded-lg hover:bg-[#EA580C] transition-colors"
        >
          <Upload className="w-5 h-5" />
          Upload Image
        </button>
        <button
          onClick={() => cameraInputRef.current?.click()}
          className="flex items-center justify-center gap-2 px-4 py-3 bg-white text-[#F97316] border-2 border-[#F97316] rounded-lg hover:bg-[#FFF7ED] transition-colors"
        >
          <Camera className="w-5 h-5" />
          Use Camera
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="hidden"
        />
        <input
          ref={cameraInputRef}
          type="file"
          accept="image/*"
          capture="environment"
          onChange={handleFileChange}
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
            disabled={isDetecting}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-[#F97316] text-white rounded-lg hover:bg-[#EA580C] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isDetecting ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Detecting...
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
  );
}
