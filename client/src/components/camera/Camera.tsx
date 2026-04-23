"use client";

import { forwardRef } from "react";

interface CameraProps {
  isStreaming: boolean;
}

const Camera = forwardRef<HTMLVideoElement, CameraProps>(({ isStreaming }, ref) => {
  return (
    <div className="relative w-full aspect-video bg-card rounded-lg overflow-hidden border border-border">
      {/* Corner bracket — top-left */}
      <span className="absolute top-2 left-2 w-5 h-5 border-t-2 border-l-2 border-[var(--color-brand)] rounded-tl z-10 pointer-events-none" />
      {/* Corner bracket — top-right */}
      <span className="absolute top-2 right-2 w-5 h-5 border-t-2 border-r-2 border-[var(--color-brand)] rounded-tr z-10 pointer-events-none" />
      {/* Corner bracket — bottom-left */}
      <span className="absolute bottom-2 left-2 w-5 h-5 border-b-2 border-l-2 border-[var(--color-brand)] rounded-bl z-10 pointer-events-none" />
      {/* Corner bracket — bottom-right */}
      <span className="absolute bottom-2 right-2 w-5 h-5 border-b-2 border-r-2 border-[var(--color-brand)] rounded-br z-10 pointer-events-none" />

      <video
        ref={ref}
        autoPlay
        playsInline
        muted
        className={`w-full h-full object-cover transition-opacity duration-300 ${isStreaming ? "opacity-100" : "opacity-0"}`}
      />

      {!isStreaming && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-muted-foreground">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
            <circle cx="12" cy="13" r="4"/>
          </svg>
          <p className="text-sm font-mono">Camera feed will appear here</p>
        </div>
      )}
    </div>
  );
});

Camera.displayName = "Camera";
export default Camera;