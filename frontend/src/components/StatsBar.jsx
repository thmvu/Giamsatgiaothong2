const LIGHT_LABEL = { red: 'ĐỎ', green: 'XANH', yellow: 'VÀNG', unknown: '--' }
const LIGHT_CLASS  = {
  red:     'text-rose-600 bg-rose-50',
  green:   'text-emerald-600 bg-emerald-50',
  yellow:  'text-amber-600 bg-amber-50',
  unknown: 'text-slate-400 bg-slate-50',
}

export default function StatsBar({ stats }) {
  const stopLineOk = stats.stop_line_ok
  const lightCls   = LIGHT_CLASS[stats.light] || LIGHT_CLASS.unknown

  const items = [
    {
      value: stats.total ? `${stats.frame ?? 0}/${stats.total}` : '—',
      label: 'Frame',
      cls: 'text-slate-700',
    },
    {
      value: LIGHT_LABEL[stats.light] || '—',
      label: 'Đèn',
      cls: lightCls,
    },
    {
      value: stopLineOk === true ? '✓' : stopLineOk === false ? '✗' : '—',
      label: 'Vạch dừng',
      cls: stopLineOk === true
        ? 'text-emerald-600 bg-emerald-50'
        : stopLineOk === false
        ? 'text-rose-500 bg-rose-50'
        : 'text-slate-400 bg-slate-50',
    },
    {
      value: stats.violations ?? 0,
      label: 'Vi phạm',
      cls: (stats.violations ?? 0) > 0 ? 'text-rose-600 bg-rose-50' : 'text-slate-700',
    },
    {
      value: stats.fps || '—',
      label: 'FPS',
      cls: 'text-slate-700',
    },
  ]

  return (
    <div className="flex border-t border-slate-200 divide-x divide-slate-200 bg-white flex-shrink-0">
      {items.map((it, i) => (
        <div key={i} className="flex-1 flex flex-col items-center justify-center py-2.5 px-1">
          <span className={`text-sm font-semibold px-1.5 py-0.5 rounded ${it.cls}`}>
            {it.value}
          </span>
          <span className="text-[10px] text-slate-400 uppercase tracking-wide mt-0.5">
            {it.label}
          </span>
        </div>
      ))}
    </div>
  )
}
