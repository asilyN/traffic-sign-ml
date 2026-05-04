'use client';

import type { CSSProperties } from 'react';
import Image from 'next/image';
import { useRouter } from 'next/navigation';
import { ArrowRight, Target, Zap } from 'lucide-react';

import { Badge } from '../ui/badge';
import {
  BRAND_ORANGE,
  FLOAT_BADGES,
  FONT_INTER,
  FONT_SYNE,
  HERO_PILL_STYLES,
  HERO_PILLS,
  MASCOT_CORNER_BRACKETS,
  MID_RING_TICKS,
} from '@/lib/landing-page';
import { cn } from '@/lib/utils';
import mascotImage from '@/src/images/traffic-sign-mascot.png';

export function LandingPage() {
  const router = useRouter();

  return (
    <div
      className="flex min-h-dvh flex-col bg-white text-slate-900 antialiased"
      style={{ fontFamily: FONT_INTER }}
    >
      {/* ══ HEADER ══ */}
      <header className="shrink-0 border-b border-slate-100">
        <div className="flex w-full items-center justify-between px-4 py-3 sm:px-6 lg:pl-8 lg:pr-10">
          <p className="text-xl font-bold tracking-tight sm:text-2xl" style={{ fontFamily: FONT_SYNE }}>
            <span className="text-slate-900">Traffic</span>
            <span style={{ color: BRAND_ORANGE }}>Scan</span>
          </p>
          <p className="flex items-center gap-2 text-xs text-slate-500 sm:text-sm">
            <span className="size-2 shrink-0 rounded-full scan-pulse" style={{ backgroundColor: BRAND_ORANGE }} />
            Real-time detection
          </p>
        </div>
      </header>

      {/* ══ MAIN ══ */}
      <main className="mx-auto flex w-full max-w-7xl flex-1 items-center px-4 py-6 sm:px-6 sm:py-8 lg:px-8 lg:py-10">
        <div className="grid w-full items-center gap-6 sm:gap-8 lg:grid-cols-2 lg:gap-14">
          {/* ── copy (below mascot on mobile, left on desktop) ── */}
          <div className="order-2 text-center lg:order-1 lg:text-left">
            <p
              className="mb-2 text-xs font-semibold uppercase tracking-[0.2em] sm:mb-3 sm:text-sm"
              style={{ color: BRAND_ORANGE, fontFamily: FONT_SYNE }}
            >
              AI-POWERED
            </p>

            <h1
              className="mb-3 text-[2.4rem] font-bold leading-[1.06] tracking-tight sm:mb-4 sm:text-5xl md:text-6xl lg:text-5xl xl:text-6xl 2xl:text-7xl"
              style={{ fontFamily: FONT_SYNE }}
            >
              <span className="block text-slate-900">Traffic Sign</span>
              <span className="block" style={{ color: BRAND_ORANGE }}>Detector</span>
            </h1>

            <p className="mx-auto mb-5 max-w-lg text-sm leading-relaxed text-slate-500 sm:mb-6 sm:text-base lg:mx-0 lg:text-lg xl:text-xl">
            Don't let a "No Left Turn" turn into a "Wrong Way" ticket. Get instant clarity before you even hit the brake. See it. Carmine it. Drive it.
            </p>

            <button
              type="button"
              onClick={() => router.push('/detect')}
              className="mb-5 inline-flex items-center gap-2 rounded-xl px-6 py-3 text-sm font-semibold text-white shadow-lg transition hover:opacity-95 focus-visible:outline-2 focus-visible:outline-offset-2 sm:mb-6 sm:gap-2.5 sm:px-8 sm:py-3.5 sm:text-base lg:px-10 lg:py-4 lg:text-lg"
              style={{ backgroundColor: BRAND_ORANGE, outlineColor: BRAND_ORANGE, fontFamily: FONT_SYNE }}
              aria-label="Open the camera detection view"
            >
              Launch Camera
              <ArrowRight className="size-5 shrink-0 sm:size-6" aria-hidden />
            </button>

            <ul className="flex flex-wrap items-center justify-center gap-2 lg:justify-start">
              {HERO_PILLS.map(({ label, icon: Icon, tone }) => {
                const styles = HERO_PILL_STYLES[tone];
                return (
                  <li
                    key={label}
                    className={cn(
                      'inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-xs font-medium sm:px-3 sm:py-1 sm:text-[0.8125rem]',
                      styles.pill,
                    )}
                  >
                    <Icon className={cn('size-3 shrink-0 sm:size-3.5', styles.icon)} aria-hidden />
                    {label}
                  </li>
                );
              })}
            </ul>
          </div>

          {/* ── mascot showcase ── */}
          <div className="order-1 flex justify-center lg:order-2 lg:justify-end">
            <div className="mascot-stage">
              {/* dot-grid background */}
              <div
                className="absolute inset-0 overflow-hidden rounded-3xl"
                style={{ background: 'radial-gradient(ellipse 80% 70% at 50% 50%, #fff5ed 0%, #fff 70%)' }}
              >
                <svg width="100%" height="100%" xmlns="http://www.w3.org/2000/svg" style={{ opacity: 0.35 }}>
                  <defs>
                    <pattern id="dots" x="0" y="0" width="24" height="24" patternUnits="userSpaceOnUse">
                      <circle cx="2" cy="2" r="1.5" fill={BRAND_ORANGE} />
                    </pattern>
                  </defs>
                  <rect width="100%" height="100%" fill="url(#dots)" />
                </svg>
                <div
                  className="absolute inset-0"
                  style={{ background: `radial-gradient(ellipse 55% 55% at 50% 55%, ${BRAND_ORANGE}22 0%, transparent 70%)` }}
                />
              </div>

              {/* corner brackets */}
              {MASCOT_CORNER_BRACKETS.map((pos, i) => {
                const { rotate, ...cornerInset } = pos;
                return (
                  <svg
                    key={i}
                    width="20"
                    height="20"
                    viewBox="0 0 28 28"
                    style={{
                      position: 'absolute',
                      transform: `rotate(${rotate}deg)`,
                      opacity: 0.45,
                      zIndex: 2,
                      ...cornerInset,
                    }}
                  >
                    <path d="M2 26 L2 2 L26 2" fill="none" stroke={BRAND_ORANGE} strokeWidth="2.5" strokeLinecap="round" />
                  </svg>
                );
              })}

              {/* outer dashed ring */}
              <div className="ring-outer">
                <svg viewBox="0 0 420 420" width="100%" height="100%">
                  <circle cx="210" cy="210" r="200" fill="none" stroke={BRAND_ORANGE} strokeWidth="1.5" strokeDasharray="8 14" opacity="0.28" />
                </svg>
              </div>

              {/* mid ring with ticks */}
              <div className="ring-mid">
                <svg viewBox="0 0 320 320" width="100%" height="100%">
                  <circle cx="160" cy="160" r="152" fill="none" stroke={BRAND_ORANGE} strokeWidth="2" strokeDasharray="4 28" opacity="0.18" />
                  {MID_RING_TICKS.map((co, i) => (
                    <line
                      key={i}
                      x1={co.x1}
                      y1={co.y1}
                      x2={co.x2}
                      y2={co.y2}
                      stroke={BRAND_ORANGE}
                      strokeWidth="2.5"
                      opacity="0.35"
                    />
                  ))}
                </svg>
              </div>

              {/* inner glow ring */}
              <div
                className="ring-inner"
                style={{
                  background: `radial-gradient(circle, ${BRAND_ORANGE}14 0%, transparent 70%)`,
                  border: `1.5px solid ${BRAND_ORANGE}30`,
                }}
              />

              {/* scan sweep */}
              <div className="scan-circle">
                <div
                  style={{
                    position: 'absolute',
                    inset: 0,
                    background: `linear-gradient(180deg, transparent 45%, ${BRAND_ORANGE}18 50%, transparent 55%)`,
                    animation: 'landing-scan-sweep 4s linear infinite',
                  }}
                />
              </div>

              {/* floating sign badges */}
              {FLOAT_BADGES.map(({ icon: Icon, label, size, delay, ...pos }) => (
                <div
                  key={label}
                  className="badge-float absolute z-20 flex flex-col items-center"
                  style={{ animationDelay: delay, ...pos } as CSSProperties}
                >
                  <Badge
                    variant="outline"
                    className={cn(
                      'h-auto min-h-0 rounded-xl border bg-white shadow-md [&>svg]:shrink-0',
                      size === 'lg' && 'gap-1 px-2.5 py-1.5 [&>svg]:size-4',
                      size === 'md' && 'gap-1 px-2 py-1 [&>svg]:size-3.5',
                      size === 'sm' && 'gap-0.5 px-1.5 py-0.5 [&>svg]:size-3',
                    )}
                    style={{ borderColor: `${BRAND_ORANGE}40` }}
                  >
                    <Icon aria-hidden style={{ color: BRAND_ORANGE }} />
                    <span className="badge-text font-bold tracking-wide" style={{ color: BRAND_ORANGE }}>
                      {label}
                    </span>
                  </Badge>
                  <div className="mt-0.5 size-1 rounded-full" style={{ backgroundColor: `${BRAND_ORANGE}60` }} />
                </div>
              ))}

              {/* stat chip – accuracy */}
              <div
                className="stat-chip absolute z-20 items-center gap-1.5 rounded-2xl border bg-white px-2.5 py-1.5 shadow-lg sm:gap-2 sm:px-3 sm:py-2"
                style={{ top: '13%', left: '-1%', borderColor: `${BRAND_ORANGE}30` }}
              >
                <div className="flex size-6 items-center justify-center rounded-full sm:size-7" style={{ backgroundColor: `${BRAND_ORANGE}18` }}>
                  <Target className="size-3.5 sm:size-4" style={{ color: BRAND_ORANGE }} />
                </div>
                <div>
                  <p className="text-[10px] leading-none text-slate-400 sm:text-xs">Accuracy</p>
                  <p className="text-xs font-bold sm:text-sm" style={{ color: BRAND_ORANGE }}>97%</p>
                </div>
              </div>

              {/* stat chip – speed */}
              <div
                className="stat-chip absolute z-20 items-center gap-1.5 rounded-2xl border bg-white px-2.5 py-1.5 shadow-lg sm:gap-2 sm:px-3 sm:py-2"
                style={{ bottom: '14%', right: '-1%', borderColor: `${BRAND_ORANGE}30` }}
              >
                <div className="flex size-6 items-center justify-center rounded-full sm:size-7" style={{ backgroundColor: `${BRAND_ORANGE}18` }}>
                  <Zap className="size-3.5 sm:size-4" style={{ color: BRAND_ORANGE }} />
                </div>
                <div>
                  <p className="text-[10px] leading-none text-slate-400 sm:text-xs">Speed</p>
                  <p className="text-xs font-bold sm:text-sm" style={{ color: BRAND_ORANGE }}>&lt;0.5s</p>
                </div>
              </div>

              {/* mascot */}
              <div className="relative z-10 flex h-full items-center justify-center">
                <Image
                  src={mascotImage}
                  alt="Traffic Sign Detector mascot"
                  className="h-auto w-auto object-contain drop-shadow-xl"
                  style={{ maxHeight: '82%', maxWidth: '68%' }}
                  width={768}
                  height={768}
                  priority
                  sizes="(min-width: 1024px) 480px, (min-width: 640px) 380px, 260px"
                />
                <div
                  className="absolute bottom-[9%] left-1/2 -translate-x-1/2 rounded-full"
                  style={{ width: '38%', height: 12, background: `radial-gradient(ellipse, ${BRAND_ORANGE}30 0%, transparent 70%)` }}
                />
              </div>

              {/* SCANNING pill */}
              <div
                className="absolute bottom-3 left-1/2 z-20 -translate-x-1/2 flex items-center gap-2 rounded-full px-3.5 py-1.5 text-xs font-bold uppercase tracking-wide text-white shadow-xl sm:bottom-4 sm:gap-2.5 sm:px-5 sm:py-2.5 sm:text-sm"
                style={{ backgroundColor: BRAND_ORANGE, fontFamily: FONT_SYNE, whiteSpace: 'nowrap' }}
              >
                <span className="size-2 shrink-0 rounded-full bg-white scan-pulse" aria-hidden />
                Scanning
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
