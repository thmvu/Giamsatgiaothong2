export default function Header({ status }) {
  const statusMap = {
    idle:        { dot: '⚪', text: 'Chờ video', color: 'var(--muted)' },
    uploading:   { dot: '🔵', text: 'Đang upload...', color: '#38bdf8' },
    calibrating: { dot: '🟡', text: 'Đang calibrate...', color: 'var(--yellow)' },
    processing:  { dot: '🟢', text: 'Đang xử lý', color: 'var(--green)' },
    done:        { dot: '✅', text: 'Hoàn tất', color: 'var(--green)' },
  }
  const s = statusMap[status] || statusMap.idle
  return (
    <header style={{
      background: 'linear-gradient(135deg,#0c1628,#0a1f3d)',
      borderBottom: '1px solid var(--border)',
      padding: '14px 24px',
      display: 'flex',
      alignItems: 'center',
      gap: 12,
      boxShadow: '0 4px 24px rgba(0,212,255,.08)',
    }}>
      <span style={{ fontSize: 26 }}>🚦</span>
      <h1 style={{
        fontSize: 20, fontWeight: 700,
        background: 'linear-gradient(90deg,#00d4ff,#7c3aed)',
        WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
      }}>AI Traffic Monitor</h1>
      <span style={{
        background: 'rgba(0,212,255,.1)', border: '1px solid rgba(0,212,255,.3)',
        color: 'var(--accent)', fontSize: 11, padding: '2px 8px', borderRadius: 20, fontWeight: 500,
      }}>FastAPI · WebSocket · YOLO</span>
      <span style={{ marginLeft: 'auto', fontSize: 13, color: s.color }}>
        {s.dot} {s.text}
      </span>
    </header>
  )
}
