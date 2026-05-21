export default function ViolationPanel({ violations }) {
  return (
    <div style={{
      background: 'var(--surface)', borderLeft: '1px solid var(--border)',
      display: 'flex', flexDirection: 'column', overflow: 'hidden',
    }}>
      <div style={{ padding: '12px 14px 8px', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', gap: 6 }}>
        {violations.length > 0 && (
          <span style={{
            display: 'inline-block', width: 8, height: 8, borderRadius: '50%',
            background: 'var(--red)', animation: 'pulse 1s infinite',
          }} />
        )}
        <span style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: 1, color: 'var(--muted)', fontWeight: 600 }}>
          Vi phạm
        </span>
        <span style={{
          marginLeft: 'auto', background: violations.length > 0 ? 'rgba(255,68,68,.15)' : 'var(--card)',
          color: violations.length > 0 ? 'var(--red)' : 'var(--muted)',
          fontSize: 11, padding: '1px 7px', borderRadius: 20, fontWeight: 600,
        }}>{violations.length}</span>
      </div>

      <div style={{ flex: 1, overflowY: 'auto', padding: '8px 10px' }}>
        {violations.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '32px 0', color: 'var(--muted)', fontSize: 13 }}>
            <div style={{ fontSize: 36, marginBottom: 8 }}>🛡️</div>
            Chưa phát hiện vi phạm
          </div>
        ) : violations.map((v, i) => (
          <VioCard key={`${v.track_id}-${v.frame}-${i}`} v={v} />
        ))}
      </div>
    </div>
  )
}

function VioCard({ v }) {
  const isHelmet = v.type === 'Không đội mũ'
  const borderColor = isHelmet ? 'var(--yellow)' : 'var(--red)'
  const typeColor   = isHelmet ? 'var(--yellow)' : 'var(--red)'
  const icon        = isHelmet ? '🪖' : '🔴'
  return (
    <div style={{
      background: 'var(--card)', border: `1px solid var(--border)`,
      borderLeft: `3px solid ${borderColor}`,
      borderRadius: 8, padding: 10, marginBottom: 8,
      animation: 'slideIn .3s ease',
    }}>
      <div style={{ fontSize: 11, fontWeight: 600, color: typeColor }}>{icon} {v.type}</div>
      <div style={{ fontSize: 11, color: 'var(--muted)', marginTop: 3 }}>
        {v.vehicle} · ID{v.track_id} · {v.time}s
      </div>

      {/* Biển số xe — hiển thị nổi bật */}
      {v.plate ? (
        <div style={{
          marginTop: 6,
          display: 'inline-flex', alignItems: 'center', gap: 5,
          background: 'rgba(14,165,233,.15)',
          border: '1px solid rgba(14,165,233,.4)',
          borderRadius: 6, padding: '3px 8px',
        }}>
          <span style={{ fontSize: 10, color: 'var(--accent)', fontWeight: 600 }}>BSX</span>
          <span style={{
            fontSize: 13, fontWeight: 800, color: '#fff',
            letterSpacing: 2, fontFamily: '"Courier New", monospace',
          }}>{v.plate}</span>
        </div>
      ) : (
        <div style={{
          marginTop: 6,
          display: 'inline-flex', alignItems: 'center', gap: 5,
          background: 'rgba(148,163,184,.08)',
          border: '1px dashed rgba(148,163,184,.25)',
          borderRadius: 6, padding: '3px 8px',
        }}>
          <span style={{ fontSize: 10, color: 'var(--muted)' }}>BSX</span>
          <span style={{
            fontSize: 11, color: 'var(--muted)', fontStyle: 'italic',
          }}>Đang nhận dạng...</span>
        </div>
      )}
    </div>
  )
}

