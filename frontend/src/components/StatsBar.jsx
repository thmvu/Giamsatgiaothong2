export default function StatsBar({ stats }) {
  const lightLabel = { red: 'ĐỎ', green: 'XANH', yellow: 'VÀNG', unknown: '--' }
  const lightColor = { red: 'var(--red)', green: 'var(--green)', yellow: 'var(--yellow)', unknown: 'var(--muted)' }
  const stopLineOk = stats.stop_line_ok
  const items = [
    { num: stats.frame ? `${stats.frame}/${stats.total}` : '0', label: 'Frame', color: 'var(--accent)' },
    { num: lightLabel[stats.light] || '--', label: 'Đèn', color: lightColor[stats.light] || 'var(--muted)' },
    {
      num: stopLineOk === true ? '✅' : stopLineOk === false ? '❌' : '--',
      label: 'Vạch dừng',
      color: stopLineOk === true ? 'var(--green)' : stopLineOk === false ? 'var(--red)' : 'var(--muted)',
    },
    { num: stats.violations, label: 'Vi phạm', color: stats.violations > 0 ? 'var(--red)' : 'var(--accent)' },
    { num: stats.fps || '--', label: 'FPS hiển thị', color: 'var(--accent)' },
  ]
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5,1fr)', borderTop: '1px solid var(--border)' }}>
      {items.map((it, i) => (
        <div key={i} style={{
          background: 'var(--surface)',
          padding: '10px 8px',
          textAlign: 'center',
          borderLeft: i > 0 ? '1px solid var(--border)' : 'none',
        }}>
          <div style={{ fontSize: 18, fontWeight: 700, color: it.color }}>{it.num}</div>
          <div style={{ fontSize: 10, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '.5px' }}>{it.label}</div>
        </div>
      ))}
    </div>
  )
}

