'use client';

import { useRouter } from 'next/navigation';
import { Zap, Target, Camera, ArrowRight } from 'lucide-react';

export function LandingPage() {
  const router = useRouter();

  return (
    <div className="min-h-screen bg-[#FFFFFF] flex flex-col items-center justify-center px-4 py-16">
      <div className="max-w-4xl w-full text-center">
        <div className="inline-flex items-center gap-2 px-4 py-2 bg-[#FEF3C7] rounded-full mb-8">
          <div className="w-2 h-2 bg-[#F97316] rounded-full"></div>
          <span className="text-[#92400E] uppercase tracking-wide">Real-time Detection</span>
        </div>

        <h1 className="mb-6">
          <span className="block text-[#6B7280] mb-2">Traffic Sign</span>
          <span className="block text-[#F97316]">Detector</span>
        </h1>

        <p className="text-[#6B7280] max-w-2xl mx-auto mb-8">
          Point your camera at any traffic sign and get an instant AI-powered classification.
        </p>

        <button
          onClick={() => router.push('/detect')}
          className="inline-flex items-center gap-2 px-8 py-4 bg-[#F97316] text-white rounded-lg hover:bg-[#EA580C] transition-colors shadow-lg hover:shadow-xl mb-16"
        >
          Launch Camera
          <ArrowRight className="w-5 h-5" />
        </button>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 max-w-3xl mx-auto">
          <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow">
            <div className="w-12 h-12 bg-[#FEF3C7] rounded-lg flex items-center justify-center mx-auto mb-4">
              <Zap className="w-6 h-6 text-[#F97316]" />
            </div>
            <h3 className="text-[#111827] mb-2">Instant</h3>
            <p className="text-[#6B7280]">Real-time</p>
          </div>

          <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow">
            <div className="w-12 h-12 bg-[#DBEAFE] rounded-lg flex items-center justify-center mx-auto mb-4">
              <Target className="w-6 h-6 text-[#2563EB]" />
            </div>
            <h3 className="text-[#111827] mb-2">Accurate</h3>
            <p className="text-[#6B7280]">CNN model</p>
          </div>

          <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow">
            <div className="w-12 h-12 bg-[#E5E7EB] rounded-lg flex items-center justify-center mx-auto mb-4">
              <Camera className="w-6 h-6 text-[#6B7280]" />
            </div>
            <h3 className="text-[#111827] mb-2">Any camera</h3>
            <p className="text-[#6B7280]">Front or rear</p>
          </div>
        </div>
      </div>
    </div>
  );
}
