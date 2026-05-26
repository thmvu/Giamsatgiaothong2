import { useEffect } from 'react'

export default function ImageModal({ src, alt = 'Ảnh bằng chứng', onClose }) {
  // Đóng bằng phím ESC
  useEffect(() => {
    const handleKey = (e) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [onClose])

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="relative bg-white rounded-2xl shadow-2xl overflow-hidden max-w-3xl w-full animate-slide-up"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-slate-100">
          <span className="text-sm font-semibold text-slate-700">{alt}</span>
          <div className="flex items-center gap-2">
            <a
              href={src}
              download
              className="text-xs text-sky-600 hover:text-sky-700 font-medium border border-sky-200 bg-sky-50
                         hover:bg-sky-100 px-3 py-1 rounded-lg transition-colors"
              onClick={e => e.stopPropagation()}
            >
              ↓ Tải xuống
            </a>
            <button
              onClick={onClose}
              className="w-7 h-7 flex items-center justify-center rounded-lg text-slate-400
                         hover:text-slate-600 hover:bg-slate-100 transition-colors text-lg leading-none"
            >
              ×
            </button>
          </div>
        </div>

        {/* Image */}
        <div className="bg-slate-50 p-2">
          <img
            src={src}
            alt={alt}
            className="w-full h-auto max-h-[75vh] object-contain rounded-lg"
          />
        </div>
      </div>
    </div>
  )
}
