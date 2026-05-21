import { useRef, useEffect } from 'react'

const LOG_COLORS = {
  vach_ok:   { bg: 'rgba(34,197,94,.12)',  border: '#22c55e', text: '#4ade80' },
  vach_fail: { bg: 'rgba(234,179,8,.10)',  border: '#eab308', text: '#fbbf24' },
  violation: { bg: 'rgba(239,68,68,.12)',  border: '#ef4444', text: '#f87171' },
  done:      { bg: 'rgba(14,165,233,.10)', border: '#0ea5e9', text: '#38bdf8' },
}

export default function VideoStream({ canvasRef, status, calibImg, logs = [] }) {
  const isActive = status === 'processing' || status === 'calibrating'
  const logEndRef = useRef(null)

  // Auto-scroll terminal to top (newest entry) — logs được prepend nên index 0 là mới nhất
  useEffect(() => {
    if (logEndRef.current) {
      logEndRef.current.scrollTop = 0
    }
  }, [logs])

  return (
    <div style={{
      flex: 1,
      display: 'flex',
      flexDirection: 'column',
      background: 'radial-gradient(ellipse at center, #0c1628 0%, #060d1a 100%)',
      overflow: 'hidden',
    }}>
      {/* ── Video area ── */}
      <div style={{
        flex: 1,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        position: 'relative',
        overflow: 'hidden',
        minHeight: 0,
      }}>
        {/* Main video canvas */}
        <canvas ref={canvasRef} style={{
          maxWidth: '100%', maxHeight: '100%',
          borderRadius: 8,
          boxShadow: isActive ? '0 0 40px rgba(0,212,255,.15)' : 'none',
          display: status === 'idle' || status === 'uploading' ? 'none' : 'block',
          animation: isActive ? 'glow 2s infinite' : 'none',
        }} />

        {/* Placeholder */}
        {(status === 'idle' || status === 'uploading') && (
          <div style={{ textAlign: 'center', color: 'var(--muted)' }}>
            <div style={{ fontSize: 72, marginBottom: 16 }}>🎥</div>
            <p style={{ fontSize: 16, fontWeight: 500, color: 'var(--text)' }}>
              {status === 'uploading' ? '📤 Đang upload video...' : 'Upload video để bắt đầu'}
            </p>
            <p style={{ fontSize: 13, marginTop: 8 }}>MP4 · AVI · MOV</p>
          </div>
        )}

        {/* Calibration overlay badge */}
        {status === 'calibrating' && (
          <div style={{
            position: 'absolute', top: 12, left: 12,
            background: 'rgba(255,215,0,.15)', border: '1px solid var(--yellow)',
            color: 'var(--yellow)', fontSize: 12, padding: '4px 10px', borderRadius: 6,
          }}>
            ⚙️ Đang calibrate vạch dừng...
          </div>
        )}

        {/* Calib preview thumbnail */}
        {calibImg && (
          <div style={{
            position: 'absolute', bottom: 12, right: 12,
            background: 'rgba(0,0,0,.6)', borderRadius: 8, padding: 6,
            border: '1px solid rgba(0,212,255,.3)',
          }}>
            <img src={calibImg} alt="calib" style={{ width: 140, borderRadius: 6 }} />
            <p style={{ fontSize: 10, color: 'var(--accent)', textAlign: 'center', marginTop: 4 }}>
              ✅ Vạch dừng đã phát hiện
            </p>
          </div>
        )}
      </div>

      {/* ── Terminal Log Panel ── */}
      <div style={{
        flexShrink: 0,
        height: logs.length > 0 ? 130 : 38,
        background: 'rgba(0,0,0,.85)',
        borderTop: '1px solid rgba(255,255,255,.07)',
        transition: 'height .3s ease',
        display: 'flex',
        flexDirection: 'column',
      }}>
        {/* Header */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 6,
          padding: '5px 10px',
          borderBottom: logs.length > 0 ? '1px solid rgba(255,255,255,.06)' : 'none',
          flexShrink: 0,
        }}>
          <span style={{
            width: 7, height: 7, borderRadius: '50%',
            background: logs.length > 0 ? '#22c55e' : '#475569',
            display: 'inline-block',
            boxShadow: logs.length > 0 ? '0 0 6px #22c55e' : 'none',
          }} />
          <span style={{ fontSize: 10, color: '#64748b', fontFamily: 'monospace', letterSpacing: '.5px', textTransform: 'uppercase' }}>
            System Log
          </span>
          {logs.length > 0 && (
            <span style={{ marginLeft: 'auto', fontSize: 10, color: '#334155', fontFamily: 'monospace' }}>
              {logs.length} entries
            </span>
          )}
        </div>

        {/* Log entries */}
        {logs.length > 0 && (
          <div
            ref={logEndRef}
            style={{
              flex: 1, overflowY: 'auto', padding: '4px 8px',
              fontFamily: '"Cascadia Code", "Fira Code", "Consolas", monospace',
              fontSize: 11,
              lineHeight: 1.6,
            }}
          >
            {logs.map((entry, i) => {
              const c = LOG_COLORS[entry.type] || { bg: 'transparent', border: '#475569', text: '#94a3b8' }
              return (
                <div key={i} style={{
                  display: 'flex', gap: 8, alignItems: 'flex-start',
                  padding: '2px 6px', marginBottom: 2, borderRadius: 4,
                  background: c.bg,
                  borderLeft: `2px solid ${c.border}`,
                }}>
                  <span style={{ color: '#475569', flexShrink: 0, fontSize: 10, paddingTop: 1 }}>
                    {entry.ts}
                  </span>
                  <span style={{ color: c.text, wordBreak: 'break-word' }}>
                    {entry.text}
                  </span>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}

