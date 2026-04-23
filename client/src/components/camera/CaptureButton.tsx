"use client";

interface CaptureButtonProps {
  onClick: () => void;
  disabled?: boolean;
  loading?: boolean;
}

export default function CaptureButton({ onClick, disabled, loading }: CaptureButtonProps) {
  return (
    <button
      onClick={onClick}
      disabled={disabled || loading}
      aria-label="Capture image for detection"
      className="relative w-18 h-18 rounded-full flex items-center justify-center
                 transition-transform hover:scale-105 active:scale-95
                 disabled:opacity-40 disabled:cursor-not-allowed disabled:transform-none
                 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-brand)]"
    >
      {/* Outer ring */}
      <span className="absolute inset-0 rounded-full border-[3px] border-[var(--color-brand)]" />

      {/* Inner circle */}
      <span className="w-13 h-13 rounded-full bg-[var(--color-brand)] flex items-center justify-center">
        {loading ? (
          <span className="w-5 h-5 rounded-full border-2 border-foreground/20 border-t-foreground animate-spin" />
        ) : (
          <span className="w-5 h-5 rounded-full bg-background" />
        )}
      </span>
    </button>
  );
}