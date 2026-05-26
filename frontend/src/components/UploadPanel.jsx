import { useRef, useState } from 'react'

const DEFAULT_CFG = {
  conf_light: 0.5,
  conf_vehicle: 0.3,
  conf_helmet: 0.4,
  conf_lp: 0.15,
  conf_ocr: 0.1,
  conf_stop_line: 0.3,
  process_every_n: 2,
  traffic_interval: 5,
  display_every_n: 3,
  display_width: 640,
  yolo_imgsz: 640,
  lp_imgsz: 1024,
  min_width_pct: 8,
  stop_line_extend_left: 150,
  stop_line_extend_right: 150,
  stop_line_offset_up: 30,
  check_redlight: true,
  check_helmet: true,
  check_plate: true,
  show_all: false,
}

/* ── Sub-components ── */
function RangeField({ label, value, min, max, step, onChange, help, fmt }) {
  const display = fmt ? fmt(value) : value
  return (
    <div className="mb-3 last:mb-0">
      <div className="flex justify-between items-center mb-1">
        <span className="text-xs font-medium text-slate-700">{label}</span>
        <span className="text-xs font-semibold text-sky-600 tabular-nums">{display}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={onChange} />
      {help && <p className="mt-1 text-[11px] text-slate-400 leading-relaxed">{help}</p>}
    </div>
  )
}

function SelectField({ label, value, onChange, help, children }) {
  return (
    <div className="mb-3 last:mb-0">
      <label className="block text-xs font-medium text-slate-700 mb-1">{label}</label>
      <select
        value={value}
        onChange={onChange}
        className="w-full text-xs bg-white border border-slate-200 rounded-lg px-3 py-1.5 text-slate-700
                   focus:outline-none focus:ring-2 focus:ring-sky-500/30 focus:border-sky-400"
      >
        {children}
      </select>
      {help && <p className="mt-1 text-[11px] text-slate-400 leading-relaxed">{help}</p>}
    </div>
  )
}

function Toggle({ label, checked, onChange }) {
  return (
    <label className="flex items-center gap-2.5 cursor-pointer py-1 group">
      <div className="relative">
        <input type="checkbox" className="sr-only peer" checked={checked} onChange={onChange} />
        <div className="w-8 h-4 bg-slate-200 rounded-full peer-checked:bg-sky-500 transition-colors" />
        <div className="absolute top-0.5 left-0.5 w-3 h-3 bg-white rounded-full shadow transition-transform peer-checked:translate-x-4" />
      </div>
      <span className="text-xs text-slate-700 group-hover:text-slate-900">{label}</span>
    </label>
  )
}

function Accordion({ title, defaultOpen = false, children }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="border border-slate-200 rounded-xl overflow-hidden mb-3">
      <button
        className="w-full flex items-center justify-between px-3.5 py-2.5 bg-white hover:bg-slate-50 transition-colors"
        onClick={() => setOpen(o => !o)}
      >
        <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">{title}</span>
        <span className={`text-slate-400 text-xs transition-transform duration-200 ${open ? 'rotate-180' : ''}`}>▼</span>
      </button>
      {open && (
        <div className="px-3.5 pt-2 pb-3 bg-white border-t border-slate-100">
          {children}
        </div>
      )}
    </div>
  )
}

/* ── Main Component ── */
export default function UploadPanel({ onStart, onStop, status }) {
  const [file, setFile]     = useState(null)
  const [cfg, setCfg]       = useState(DEFAULT_CFG)
  const [dragging, setDrag] = useState(false)
  const inputRef            = useRef(null)

  const busy   = status !== 'idle' && status !== 'done'
  const setNum  = (k, v) => setCfg(p => ({ ...p, [k]: Number(v) }))
  const setBool = (k, v) => setCfg(p => ({ ...p, [k]: v }))

  return (
    <aside className="w-80 flex-shrink-0 flex flex-col bg-slate-50 border-r border-slate-200 overflow-hidden">
      <div className="flex-1 overflow-y-auto scrollbar-thin p-3">

        {/* ── Drop zone ── */}
        <div
          className={`mb-3 border-2 border-dashed rounded-xl p-5 text-center cursor-pointer transition-all
            ${dragging
              ? 'border-sky-400 bg-sky-50'
              : 'border-slate-200 bg-white hover:border-sky-300 hover:bg-sky-50/50'}`}
          onClick={() => inputRef.current?.click()}
          onDragOver={e => { e.preventDefault(); setDrag(true) }}
          onDragLeave={() => setDrag(false)}
          onDrop={e => { e.preventDefault(); setDrag(false); const f = e.dataTransfer.files?.[0]; if (f) setFile(f) }}
        >
          <div className="text-3xl mb-2">🎬</div>
          <p className="text-sm font-medium text-slate-700">Tải video giao thông</p>
          <p className="text-xs text-slate-400 mt-1">Kéo thả hoặc bấm để chọn</p>
          <p className="text-xs text-slate-400">.mp4 · .avi · .mov</p>
          {file && (
            <div className="mt-2.5 inline-flex items-center gap-1.5 bg-sky-50 border border-sky-200 text-sky-700
                            text-xs font-medium px-2.5 py-1 rounded-full max-w-full">
              <span>✓</span>
              <span className="truncate">{file.name}</span>
            </div>
          )}
        </div>
        <input
          ref={inputRef} type="file" accept=".mp4,.avi,.mov"
          className="hidden"
          onChange={e => { const f = e.target.files?.[0]; if (f) setFile(f) }}
        />

        {/* ── Accordion: Ngưỡng phát hiện ── */}
        <Accordion title="⚙️ Ngưỡng phát hiện" defaultOpen>
          <RangeField label="Đèn giao thông" value={cfg.conf_light} min="0.1" max="0.9" step="0.05"
            onChange={e => setNum('conf_light', e.target.value)} fmt={v => v.toFixed(2)} />
          <RangeField label="Phương tiện" value={cfg.conf_vehicle} min="0.1" max="0.9" step="0.05"
            onChange={e => setNum('conf_vehicle', e.target.value)} fmt={v => v.toFixed(2)} />
          <RangeField label="Mũ bảo hiểm" value={cfg.conf_helmet} min="0.1" max="0.9" step="0.05"
            onChange={e => setNum('conf_helmet', e.target.value)} fmt={v => v.toFixed(2)} />
        </Accordion>

        {/* ── Accordion: Tính năng ── */}
        <Accordion title="🔍 Tính năng" defaultOpen>
          <Toggle label="🔴 Phát hiện vượt đèn đỏ"  checked={cfg.check_redlight} onChange={e => setBool('check_redlight', e.target.checked)} />
          <Toggle label="🪖 Kiểm tra mũ bảo hiểm"   checked={cfg.check_helmet}   onChange={e => setBool('check_helmet',   e.target.checked)} />
          <Toggle label="🔢 Nhận dạng biển số (OCR)" checked={cfg.check_plate}    onChange={e => setBool('check_plate',    e.target.checked)} />
          <Toggle label="👁️ Hiện tất cả xe"          checked={cfg.show_all}       onChange={e => setBool('show_all',       e.target.checked)} />

          {cfg.check_plate && (
            <div className="mt-2 pt-2 border-t border-slate-100">
              <RangeField label="YOLO Biển số — conf" value={cfg.conf_lp} min="0.05" max="0.5" step="0.05"
                onChange={e => setNum('conf_lp', e.target.value)} fmt={v => v.toFixed(2)}
                help="Giảm nếu biển số nhỏ hoặc dễ bị bỏ sót." />
              <RangeField label="OCR — ngưỡng chấp nhận" value={cfg.conf_ocr} min="0.05" max="0.9" step="0.05"
                onChange={e => setNum('conf_ocr', e.target.value)} fmt={v => v.toFixed(2)} />
              <SelectField label="LP YOLO imgsz" value={cfg.lp_imgsz} onChange={e => setNum('lp_imgsz', e.target.value)}
                help="1024 chính xác hơn cho ONNX nặng.">
                <option value={320}>320 — rất nhanh</option>
                <option value={640}>640 — cân bằng</option>
                <option value={1024}>1024 — chính xác hơn</option>
              </SelectField>
            </div>
          )}
        </Accordion>

        {/* ── Accordion: Vạch dừng (SAM) ── */}
        <Accordion title="🚧 Định chuẩn vạch dừng">
          <RangeField label="Confidence vạch dừng" value={cfg.conf_stop_line} min="0.1" max="0.9" step="0.05"
            onChange={e => setNum('conf_stop_line', e.target.value)} fmt={v => v.toFixed(2)} />
          <RangeField label="Min width vạch (% frame)" value={cfg.min_width_pct} min="1" max="40" step="1"
            onChange={e => setNum('min_width_pct', e.target.value)}
            help="Video lớn giảm xuống 5–10%, mặc định 8%." />
          <RangeField label="Mở rộng trái (px)" value={cfg.stop_line_extend_left} min="0" max="500" step="10"
            onChange={e => setNum('stop_line_extend_left', e.target.value)} />
          <RangeField label="Mở rộng phải (px)" value={cfg.stop_line_extend_right} min="0" max="500" step="10"
            onChange={e => setNum('stop_line_extend_right', e.target.value)} />
          <RangeField label="Dịch vạch lên (px)" value={cfg.stop_line_offset_up} min="0" max="100" step="5"
            onChange={e => setNum('stop_line_offset_up', e.target.value)}
            help="Nâng ngưỡng vi phạm lên để giảm false-positive." />
        </Accordion>

        {/* ── Accordion: Hiệu năng ── */}
        <Accordion title="⚡ Hiệu năng">
          <RangeField label="Xử lý mỗi N frame (AI)" value={cfg.process_every_n} min="1" max="5" step="1"
            onChange={e => setNum('process_every_n', e.target.value)} />
          <RangeField label="Check đèn mỗi N frame" value={cfg.traffic_interval} min="1" max="10" step="1"
            onChange={e => setNum('traffic_interval', e.target.value)} />
          <RangeField label="Cập nhật UI mỗi N frame" value={cfg.display_every_n} min="1" max="10" step="1"
            onChange={e => setNum('display_every_n', e.target.value)}
            help="Tăng lên nếu demo lag trên máy yếu." />
          <SelectField label="Độ rộng hiển thị (px)" value={cfg.display_width} onChange={e => setNum('display_width', e.target.value)}>
            <option value={320}>320</option>
            <option value={480}>480</option>
            <option value={640}>640</option>
            <option value={800}>800</option>
            <option value={960}>960</option>
            <option value={1280}>1280</option>
          </SelectField>
          <SelectField label="YOLO imgsz (xe/đèn)" value={cfg.yolo_imgsz} onChange={e => setNum('yolo_imgsz', e.target.value)}>
            <option value={320}>320</option>
            <option value={480}>480</option>
            <option value={640}>640</option>
            <option value={960}>960</option>
            <option value={1280}>1280</option>
          </SelectField>
        </Accordion>

      </div>

      {/* ── Actions ── */}
      <div className="flex-shrink-0 p-3 border-t border-slate-200 bg-white space-y-2">
        <button
          id="btn-start"
          disabled={!file || busy}
          onClick={() => onStart(file, cfg)}
          className="w-full py-2.5 rounded-xl text-sm font-semibold transition-all
            bg-sky-500 text-white hover:bg-sky-600 active:scale-[.98]
            disabled:opacity-40 disabled:cursor-not-allowed disabled:active:scale-100"
        >
          {busy ? '⏳ Đang xử lý...' : '▶ Bắt đầu quét'}
        </button>
        <button
          id="btn-stop"
          disabled={!busy}
          onClick={onStop}
          className="w-full py-2 rounded-xl text-sm font-medium transition-all
            bg-rose-50 text-rose-600 border border-rose-200 hover:bg-rose-100
            disabled:opacity-40 disabled:cursor-not-allowed"
        >
          ■ Dừng
        </button>
      </div>
    </aside>
  )
}
