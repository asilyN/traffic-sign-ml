import { Upload, Loader2, Scan, RefreshCw } from 'lucide-react';

interface UploadViewProps {
  selectedImage: string | null;
  isDetecting: boolean;
  isDraggingFile: boolean;
  fileInputRef: React.RefObject<HTMLInputElement>;
  onDragOver: (e: React.DragEvent) => void;
  onDragLeave: (e: React.DragEvent) => void;
  onDrop: (e: React.DragEvent) => void;
  onFileClick: () => void;
  onFileChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onDetect: () => void;
  onClear: () => void;
}

export function UploadView({
  selectedImage,
  isDetecting,
  isDraggingFile,
  fileInputRef,
  onDragOver,
  onDragLeave,
  onDrop,
  onFileClick,
  onFileChange,
  onDetect,
  onClear,
}: UploadViewProps) {
  return (
    <div className="space-y-4">
      <input
        ref={fileInputRef}
        type="file"
        accept="image/png,image/jpeg,image/webp,image/*"
        onChange={onFileChange}
        className="hidden"
      />
      {!selectedImage ? (
        <>
          <button
            type="button"
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
            onClick={onFileClick}
            className={`relative w-full rounded-2xl border-2 border-dashed transition-colors min-h-70 flex flex-col items-center justify-center px-6 py-10 cursor-pointer bg-[#F8FAFC] hover:bg-[#F1F5F9] ${
              isDraggingFile ? 'border-[#F97316] bg-orange-50/50' : 'border-[#CBD5E1]'
            }`}
          >
            <div className="mb-4 rounded-2xl bg-[#FFF7ED] p-4 ring-1 ring-[#F97316]/20">
              <Upload className="w-10 h-10 text-[#F97316]" strokeWidth={2} />
            </div>
            <p className="text-[#0F172A] font-medium text-center mb-1">
              Drop an image here or click the button below to browse.
            </p>
            <p className="text-[#64748B] text-xs mt-4 flex items-center gap-3">
              <span className="h-px w-10 bg-[#E2E8F0]" aria-hidden />
              PNG · JPG · WEBP
              <span className="h-px w-10 bg-[#E2E8F0]" aria-hidden />
            </p>
          </button>
          <button
            type="button"
            onClick={onFileClick}
            className="w-full flex items-center justify-center gap-2 rounded-xl border-2 border-[#E2E8F0] bg-white text-[#0F172A] px-4 py-3.5 text-sm font-semibold hover:border-[#CBD5E1] hover:bg-[#FAFAFA] transition-colors"
          >
            <Upload className="w-5 h-5 text-[#64748B]" />
            Browse Files
          </button>
          <button
            type="button"
            disabled
            className="w-full flex items-center justify-center gap-2 rounded-xl bg-[#F97316]/40 text-white px-4 py-4 text-[15px] font-semibold cursor-not-allowed"
          >
            <Scan className="w-5 h-5" />
            Detect Sign
          </button>
        </>
      ) : (
        <>
          <div className="relative rounded-2xl border-2 border-[#E2E8F0] overflow-hidden bg-[#F8FAFC]">
            <button
              type="button"
              onClick={onClear}
              className="absolute top-3 right-3 z-10 inline-flex items-center gap-1.5 rounded-full bg-white/95 border border-[#E2E8F0] px-3 py-1.5 text-xs font-semibold text-[#64748B] shadow-sm hover:bg-white hover:text-[#0F172A] transition-colors"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              Reset
            </button>
            <img
              src={selectedImage}
              alt="Selected sign"
              className="w-full h-auto max-h-95 object-contain mx-auto block"
            />
          </div>
          <button
            type="button"
            onClick={onFileClick}
            className="w-full flex items-center justify-center gap-2 rounded-xl border-2 border-[#E2E8F0] bg-white text-[#0F172A] px-4 py-3.5 text-sm font-semibold hover:border-[#CBD5E1] hover:bg-[#FAFAFA] transition-colors"
          >
            <Upload className="w-5 h-5 text-[#64748B]" />
            Browse Files
          </button>
          <button
            type="button"
            onClick={onDetect}
            disabled={isDetecting}
            className="w-full flex items-center justify-center gap-2 rounded-xl bg-[#F97316] text-white px-4 py-4 text-[15px] font-semibold hover:bg-[#EA580C] disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-lg shadow-orange-500/25"
          >
            {isDetecting ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Detecting…
              </>
            ) : (
              <>
                <Scan className="w-5 h-5" />
                Detect Sign
              </>
            )}
          </button>
        </>
      )}
    </div>
  );
}
