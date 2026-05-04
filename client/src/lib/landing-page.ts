import type { LucideIcon } from 'lucide-react';
import {
  Camera,
  Gauge,
  OctagonAlert,
  School,
  Target,
  TriangleAlert,
  Undo2,
  Zap,
} from 'lucide-react';

export const BRAND_ORANGE = '#E86105';

export const FONT_SYNE = "'Syne', sans-serif";
export const FONT_INTER = "'Inter', sans-serif";

/** Precomputed so SSR/CSR SVG coords match (avoids hydration mismatch from float jitter). */
export const MID_RING_TICKS = Array.from({ length: 12 }, (_, i) => {
  const a = (i / 12) * 2 * Math.PI;
  const c = Math.cos(a);
  const s = Math.sin(a);
  const q = (n: number) => Math.round(n * 1000) / 1000;
  return {
    x1: q(160 + 146 * c),
    y1: q(160 + 146 * s),
    x2: q(160 + 158 * c),
    y2: q(160 + 158 * s),
  };
});

export type HeroPillTone = 'instant' | 'accuracy' | 'camera';

export type HeroPill = { label: string; icon: LucideIcon; tone: HeroPillTone };

/** Compact pill + icon colors aligned with TrafficScan (orange CTA + slate copy). */
export const HERO_PILL_STYLES: Record<HeroPillTone, { pill: string; icon: string }> = {
  instant: {
    pill: 'border-orange-200/90 bg-orange-50 text-slate-800',
    icon: 'text-orange-600',
  },
  accuracy: {
    pill: 'border-slate-200 bg-slate-100 text-slate-800',
    icon: 'text-[#E86105]',
  },
  camera: {
    pill: 'border-sky-200/90 bg-sky-50 text-slate-800',
    icon: 'text-sky-600',
  },
};

export const HERO_PILLS: readonly HeroPill[] = [
  { label: 'Instant', icon: Zap, tone: 'instant' },
  { label: '97% accuracy', icon: Target, tone: 'accuracy' },
  { label: 'Any camera', icon: Camera, tone: 'camera' },
];

export type FloatBadgeConfig = {
  icon: LucideIcon;
  label: string;
  top?: string;
  left?: string;
  right?: string;
  delay: string;
  size: 'sm' | 'md' | 'lg';
};

export const FLOAT_BADGES: readonly FloatBadgeConfig[] = [
  { icon: OctagonAlert, label: 'STOP', top: '8%', left: '3%', delay: '0s', size: 'lg' },
  { icon: TriangleAlert, label: 'YIELD', top: '10%', right: '3%', delay: '0.4s', size: 'sm' },
  { icon: School, label: 'SCHOOL', top: '60%', left: '1%', delay: '0.8s', size: 'sm' },
  { icon: Gauge, label: '60', top: '70%', right: '2%', delay: '1.2s', size: 'md' },
  { icon: Undo2, label: 'U-TURN', top: '36%', right: '1%', delay: '0.2s', size: 'sm' },
];

/** Corner bracket decorations around the mascot stage (px + rotation). */
export const MASCOT_CORNER_BRACKETS = [
  { top: 12, left: 12, rotate: 0 },
  { top: 12, right: 12, rotate: 90 },
  { bottom: 12, right: 12, rotate: 180 },
  { bottom: 12, left: 12, rotate: 270 },
] as const;
