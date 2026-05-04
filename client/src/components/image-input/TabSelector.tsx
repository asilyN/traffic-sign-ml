import { Camera, Upload } from 'lucide-react';

type TabId = 'camera' | 'upload';

interface TabSelectorProps {
  activeTab: TabId;
  onTabChange: (tab: TabId) => void;
}

export function TabSelector({ activeTab, onTabChange }: TabSelectorProps) {
  const tabBtn =
    'relative flex flex-1 items-center justify-center gap-2 rounded-xl px-4 py-3 text-sm font-semibold transition-all duration-200';

  return (
    <div className="rounded-2xl bg-[#EEF2F6] p-1.5 flex gap-1 shadow-inner">
      <button
        type="button"
        onClick={() => onTabChange('camera')}
        className={`${tabBtn} ${
          activeTab === 'camera'
            ? 'bg-white text-[#F97316] shadow-md shadow-black/5'
            : 'text-[#64748B] hover:text-[#0F172A]'
        }`}
      >
        <Camera className="w-4 h-4 shrink-0" />
        Camera
      </button>
      <button
        type="button"
        onClick={() => onTabChange('upload')}
        className={`${tabBtn} ${
          activeTab === 'upload'
            ? 'bg-white text-[#F97316] shadow-md shadow-black/5'
            : 'text-[#64748B] hover:text-[#0F172A]'
        }`}
      >
        <Upload className="w-4 h-4 shrink-0" />
        Upload Image
      </button>
    </div>
  );
}
