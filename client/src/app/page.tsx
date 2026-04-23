import Link from "next/link";

export default function Home() {
  return (
    <div className="relative min-h-dvh flex items-center justify-center overflow-hidden px-4 py-16">
      {/* Grid background */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          backgroundImage: `
            linear-gradient(oklch(0.85 0.18 85 / 4%) 1px, transparent 1px),
            linear-gradient(90deg, oklch(0.85 0.18 85 / 4%) 1px, transparent 1px)
          `,
          backgroundSize: "48px 48px",
          maskImage: "radial-gradient(ellipse 80% 80% at 50% 50%, black 30%, transparent 100%)",
        }}
      />

      <main className="relative flex flex-col items-center text-center gap-8 max-w-lg">
        {/* Live badge */}
        <div className="inline-flex items-center gap-2 bg-[var(--color-brand-dim)] border border-[var(--color-brand)]/30
                        text-[var(--color-brand)] text-[0.7rem] font-mono tracking-widest uppercase
                        px-4 py-1.5 rounded-full">
          <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-brand)] animate-pulse-dot" />
          Real-time detection
        </div>

        {/* Headline */}
        <div className="flex flex-col leading-none">
          <span className="text-[clamp(2.5rem,8vw,4.5rem)] font-bold tracking-tight text-muted-foreground">
            Traffic Sign
          </span>
          <span className="text-[clamp(3rem,12vw,6.5rem)] font-bold tracking-tighter text-[var(--color-brand)] leading-[0.9]">
            Detector
          </span>
        </div>

        <p className="text-muted-foreground text-base leading-relaxed max-w-sm">
          Point your camera at any traffic sign and get an instant AI-powered classification.
        </p>

        <Link
          href="/detect"
          className="inline-flex items-center gap-2.5 bg-[var(--color-brand)] text-background
                     font-bold text-base px-7 py-3.5 rounded-lg
                     transition-all hover:-translate-y-0.5 hover:shadow-[0_8px_32px_var(--color-brand-glow)]
                     active:translate-y-0"
        >
          Launch Camera
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M5 12h14M12 5l7 7-7 7"/>
          </svg>
        </Link>

        {/* Features row */}
        <div className="grid grid-cols-3 gap-3 w-full">
          {[
            { icon: "⚡", label: "Instant",   desc: "Real-time" },
            { icon: "🎯", label: "Accurate",  desc: "CNN model" },
            { icon: "📷", label: "Any camera",desc: "Front or rear" },
          ].map(({ icon, label, desc }) => (
            <div
              key={label}
              className="flex flex-col items-center gap-1 bg-card border border-border
                         rounded-lg py-4 px-2 text-xs"
            >
              <span className="text-xl mb-1">{icon}</span>
              <strong className="text-foreground text-sm">{label}</strong>
              <span className="text-muted-foreground">{desc}</span>
            </div>
          ))}
        </div>
      </main>
    </div>
  );
}