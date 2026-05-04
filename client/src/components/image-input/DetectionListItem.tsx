import { Detection } from '@/src/lib/api';

interface DetectionListItemProps {
  detection: Detection;
  index: number;
  isHovered: boolean;
  onHover: (index: number | null) => void;
}

export function DetectionListItem({
  detection: det,
  index: i,
  isHovered,
  onHover,
}: DetectionListItemProps) {
  return (
    <div
      className="relative rounded-xl border border-[#F97316]/25 bg-[#FFF7ED] p-3 text-sm hover:border-[#F97316]/50 hover:shadow-sm transition-all"
      onMouseEnter={() => onHover(i)}
      onMouseLeave={() => onHover(null)}
    >
      <div className="flex justify-between items-start gap-2">
        <div className="min-w-0">
          <p className="font-semibold text-[#0F172A] truncate">{det.class_name}</p>
          <p className="text-[#64748B] text-xs">{det.category}</p>
        </div>
        <span className="text-[#F97316] font-bold tabular-nums shrink-0">
          {(det.classification_confidence * 100).toFixed(1)}%
        </span>
      </div>
      {isHovered && det.other_predictions && det.other_predictions.length > 0 && (
        <div className="absolute left-0 right-0 top-full z-10 mt-2 rounded-xl border border-[#F97316]/30 bg-white p-3 shadow-lg space-y-1.5">
          <p className="text-[11px] font-semibold text-[#0F172A] uppercase tracking-wide">
            Top predictions
          </p>
          {[det, ...det.other_predictions].slice(0, 3).map((pred, idx) => {
            const label =
              idx === 0 ? det.class_name : (pred as (typeof det.other_predictions)[0]).class_name;
            const conf =
              idx === 0
                ? det.classification_confidence
                : (pred as (typeof det.other_predictions)[0]).confidence;
            return (
              <div key={idx} className="flex justify-between gap-2 text-xs">
                <span className="text-[#334155] truncate">
                  {idx + 1}. {label}
                </span>
                <span className="font-semibold text-[#F97316] shrink-0">
                  {(conf * 100).toFixed(1)}%
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
