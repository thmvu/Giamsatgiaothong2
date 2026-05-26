export default function ProgressBar({ progress }) {
  const pct = Math.round((progress.pct || 0) * 100)
  return (
    <div className="px-4 py-2 bg-white border-b border-slate-200 flex items-center gap-3">
      <div className="flex-1 h-1.5 bg-slate-100 rounded-full overflow-hidden">
        <div
          className="h-full bg-sky-500 rounded-full transition-all duration-300"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="flex-shrink-0 text-[11px] text-slate-500 min-w-[3rem] text-right font-medium">
        {pct}%
      </span>
      {progress.msg && (
        <span className="hidden sm:block text-[11px] text-slate-400 truncate max-w-[260px]">
          {progress.msg}
        </span>
      )}
    </div>
  )
}
