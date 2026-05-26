import { useState } from 'react'
import ImageModal from './ImageModal'

export default function ViolationPanel({ violations }) {
  const [modal, setModal] = useState(null) // { src, alt }

  return (
    <>
      <aside className="w-72 flex-shrink-0 flex flex-col bg-white border-l border-slate-200 overflow-hidden">
        {/* Header */}
        <div className="flex items-center gap-2 px-4 py-3 border-b border-slate-200 flex-shrink-0">
          {violations.length > 0 && (
            <span className="w-2 h-2 rounded-full bg-rose-400 animate-pulse-dot" />
          )}
          <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
            Vi phạm
          </span>
          <span className={`ml-auto text-[11px] font-semibold px-2 py-0.5 rounded-full
            ${violations.length > 0
              ? 'bg-rose-50 text-rose-600 border border-rose-200'
              : 'bg-slate-100 text-slate-400'}`}>
            {violations.length}
          </span>
        </div>

        {/* List */}
        <div className="flex-1 overflow-y-auto scrollbar-thin px-3 py-2 space-y-2">
          {violations.length === 0 ? (
            <div className="flex flex-col items-center gap-2 py-10 text-slate-400 select-none">
              <span className="text-4xl">🛡️</span>
              <p className="text-xs">Chưa phát hiện vi phạm</p>
            </div>
          ) : violations.map((v, i) => (
            <VioCard key={`${v.track_id}-${v.frame}-${i}`} v={v} onOpenImage={setModal} />
          ))}
        </div>
      </aside>

      {/* Modal */}
      {modal && <ImageModal src={modal.src} alt={modal.alt} onClose={() => setModal(null)} />}
    </>
  )
}

function VioCard({ v, onOpenImage }) {
  const isHelmet = v.type === 'Không đội mũ'
  const typeStyle = isHelmet
    ? { badge: 'bg-amber-50 text-amber-700 border-amber-200', bar: 'bg-amber-400', icon: '🪖' }
    : { badge: 'bg-rose-50 text-rose-700 border-rose-200',   bar: 'bg-rose-400',   icon: '🔴' }

  return (
    <div className="animate-slide-up bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm hover:shadow-md transition-shadow">
      {/* Thanh màu phân biệt loại vi phạm */}
      <div className={`h-1 w-full ${typeStyle.bar}`} />

      <div className="p-3">
        {/* Type & meta */}
        <div className="flex items-start justify-between gap-2 mb-2">
          <span className={`inline-flex items-center gap-1 text-[11px] font-semibold px-2 py-0.5 rounded-full border ${typeStyle.badge}`}>
            {typeStyle.icon} {v.type}
          </span>
          <span className="text-[10px] text-slate-400 whitespace-nowrap">{v.time}s</span>
        </div>

        <p className="text-[11px] text-slate-500 mb-2">
          {v.vehicle} · ID {v.track_id}
        </p>

        {/* Biển số xe */}
        <div className="mb-2.5">
          {v.plate ? (
            <div className="inline-flex items-center gap-2 bg-yellow-50 border-2 border-yellow-400
                            rounded-lg px-2.5 py-1">
              <span className="text-[9px] font-bold text-yellow-700 uppercase tracking-widest">BSX</span>
              <span className="text-sm font-black text-slate-800 tracking-[0.2em] font-mono">
                {v.plate}
              </span>
            </div>
          ) : (
            <div className="inline-flex items-center gap-1.5 bg-slate-50 border border-slate-200
                            rounded-lg px-2.5 py-1 text-[11px] text-slate-400">
              <span className="animate-pulse-dot">●</span>
              Đang nhận dạng biển số...
            </div>
          )}
        </div>

        {/* Ảnh bằng chứng */}
        {(v.evidenceUrl || v.cropUrl) && (
          <div className="flex gap-2">
            {v.evidenceUrl && (
              <button
                onClick={() => onOpenImage({ src: v.evidenceUrl, alt: `Bằng chứng vi phạm ID${v.track_id}` })}
                className="flex-1 group relative rounded-lg overflow-hidden border border-slate-200 hover:border-sky-400 transition-colors cursor-zoom-in"
                title="Xem ảnh bằng chứng"
              >
                <img
                  src={v.evidenceUrl}
                  alt="Bằng chứng vi phạm"
                  className="w-full h-16 object-cover"
                  onError={e => { e.target.style.display = 'none' }}
                />
                <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors flex items-center justify-center">
                  <span className="opacity-0 group-hover:opacity-100 text-white text-lg transition-opacity drop-shadow">🔍</span>
                </div>
                <span className="absolute bottom-0 inset-x-0 bg-black/40 text-[9px] text-white text-center py-0.5">
                  Vi phạm
                </span>
              </button>
            )}
            {v.cropUrl && (
              <button
                onClick={() => onOpenImage({ src: v.cropUrl, alt: `Biển số ID${v.track_id}` })}
                className="flex-1 group relative rounded-lg overflow-hidden border border-slate-200 hover:border-sky-400 transition-colors cursor-zoom-in"
                title="Xem ảnh biển số"
              >
                <img
                  src={v.cropUrl}
                  alt="Ảnh cắt biển số"
                  className="w-full h-16 object-cover"
                  onError={e => { e.target.style.display = 'none' }}
                />
                <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors flex items-center justify-center">
                  <span className="opacity-0 group-hover:opacity-100 text-white text-lg transition-opacity drop-shadow">🔍</span>
                </div>
                <span className="absolute bottom-0 inset-x-0 bg-black/40 text-[9px] text-white text-center py-0.5">
                  Biển số
                </span>
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
