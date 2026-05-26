import { useRef, useEffect } from 'react'

const LOG_STYLE = {
  vach_ok:   { bar: 'bg-emerald-400', text: 'text-emerald-700', bg: 'bg-emerald-50' },
  vach_fail: { bar: 'bg-amber-400',   text: 'text-amber-700',   bg: 'bg-amber-50'   },
  violation: { bar: 'bg-rose-400',    text: 'text-rose-700',    bg: 'bg-rose-50'    },
  done:      { bar: 'bg-sky-400',     text: 'text-sky-700',     bg: 'bg-sky-50'     },
}

export default function VideoStream({ canvasRef, status, calibImg, logs = [] }) {
  const isActive   = status === 'processing' || status === 'calibrating'
  const showCanvas = status !== 'idle' && status !== 'uploading'
  const logEndRef  = useRef(null)

  useEffect(() => {
    if (logEndRef.current) logEndRef.current.scrollTop = 0
  }, [logs])

  return (
    <div className="flex-1 flex flex-col bg-slate-50 overflow-hidden min-h-0">
      {/* ── Video area ── */}
      <div className="flex-1 flex items-center justify-center relative overflow-hidden min-h-0 p-3">

        {/* Canvas */}
        <canvas
          ref={canvasRef}
          className="max-w-full max-h-full rounded-xl border border-slate-200 shadow-sm object-contain"
          style={{ display: showCanvas ? 'block' : 'none' }}
        />

        {/* Placeholder */}
        {!showCanvas && (
          <div className="flex flex-col items-center gap-3 text-slate-400 select-none">
            <span className="text-7xl">{status === 'uploading' ? '📤' : '🎥'}</span>
            <p className="text-sm font-medium text-slate-500">
              {status === 'uploading' ? 'Đang upload video...' : 'Upload video để bắt đầu'}
            </p>
            <p className="text-xs text-slate-400">MP4 · AVI · MOV</p>
          </div>
        )}

        {/* Calibrating badge */}
        {status === 'calibrating' && (
          <div className="absolute top-5 left-5 flex items-center gap-1.5 bg-amber-50 border border-amber-200 text-amber-700 text-xs font-medium px-3 py-1.5 rounded-full shadow-sm">
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse-dot" />
            Đang calibrate vạch dừng...
          </div>
        )}

        {/* Calib preview */}
        {calibImg && (
          <div className="absolute bottom-5 right-5 bg-white border border-slate-200 rounded-xl p-2 shadow-md">
            <img src={calibImg} alt="calib" className="w-36 rounded-lg" />
            <p className="text-[10px] text-emerald-600 text-center mt-1.5 font-medium">
              ✓ Vạch dừng đã phát hiện
            </p>
          </div>
        )}
      </div>

      {/* ── Terminal Log ── */}
      <div
        className="flex-shrink-0 border-t border-slate-200 bg-white flex flex-col overflow-hidden transition-all duration-300"
        style={{ height: logs.length > 0 ? 130 : 36 }}
      >
        {/* Header bar */}
        <div className="flex items-center gap-2 px-3 py-1.5 border-b border-slate-100 flex-shrink-0">
          <span
            className={`w-1.5 h-1.5 rounded-full ${logs.length > 0 ? 'bg-emerald-400 animate-pulse-dot' : 'bg-slate-300'}`}
          />
          <span className="text-[10px] font-medium text-slate-400 uppercase tracking-widest font-mono">
            System Log
          </span>
          {logs.length > 0 && (
            <span className="ml-auto text-[10px] text-slate-300 font-mono">
              {logs.length} entries
            </span>
          )}
        </div>

        {/* Log entries */}
        {logs.length > 0 && (
          <div
            ref={logEndRef}
            className="flex-1 overflow-y-auto scrollbar-thin px-2 py-1 font-mono text-[11px] space-y-0.5"
          >
            {logs.map((entry, i) => {
              const c = LOG_STYLE[entry.type] || { bar: 'bg-slate-300', text: 'text-slate-600', bg: 'bg-slate-50' }
              return (
                <div key={i} className={`flex gap-2 items-start px-2 py-0.5 rounded ${c.bg}`}>
                  <span className={`flex-shrink-0 w-0.5 self-stretch rounded-full ${c.bar}`} />
                  <span className="text-slate-400 flex-shrink-0">{entry.ts}</span>
                  <span className={`${c.text} break-words min-w-0`}>{entry.text}</span>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
