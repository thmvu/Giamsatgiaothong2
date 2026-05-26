const STATUS_MAP = {
  idle:        { dot: 'bg-slate-300',  text: 'Chờ video',          textColor: 'text-slate-500' },
  uploading:   { dot: 'bg-sky-400 animate-pulse-dot', text: 'Đang upload...', textColor: 'text-sky-600' },
  calibrating: { dot: 'bg-amber-400 animate-pulse-dot', text: 'Đang calibrate...', textColor: 'text-amber-600' },
  processing:  { dot: 'bg-emerald-400 animate-pulse-dot', text: 'Đang xử lý', textColor: 'text-emerald-600' },
  done:        { dot: 'bg-emerald-500', text: 'Hoàn tất',         textColor: 'text-emerald-600' },
}

export default function Header({ status }) {
  const s = STATUS_MAP[status] || STATUS_MAP.idle
  return (
    <header className="h-14 flex-shrink-0 flex items-center gap-3 px-5 bg-white border-b border-slate-200 shadow-sm z-10">
      {/* Logo */}
      <div className="flex items-center gap-2">
        <span className="text-2xl leading-none">🚦</span>
        <h1 className="text-[15px] font-semibold text-slate-800 tracking-tight">
          AI Traffic Monitor
        </h1>
      </div>

      {/* Badge */}
      <span className="hidden sm:inline-flex items-center gap-1 text-[10px] font-medium text-slate-400 bg-slate-100 border border-slate-200 px-2 py-0.5 rounded-full tracking-wide">
        FastAPI · WebSocket · YOLO11
      </span>

      {/* Status */}
      <div className="ml-auto flex items-center gap-2">
        <span className={`inline-block w-2 h-2 rounded-full ${s.dot}`} />
        <span className={`text-xs font-medium ${s.textColor}`}>{s.text}</span>
      </div>
    </header>
  )
}
