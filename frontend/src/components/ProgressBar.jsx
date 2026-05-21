export default function ProgressBar({ progress }) {
  return (
    <div style={{ padding: '8px 16px', background: 'var(--surface)', borderTop: '1px solid var(--border)' }}>
      <div style={{ background: 'var(--border)', borderRadius: 4, height: 6, overflow: 'hidden' }}>
        <div style={{
          height: '100%',
          background: 'linear-gradient(90deg,var(--accent),var(--accent2))',
          borderRadius: 4,
          width: `${Math.round(progress.pct * 100)}%`,
          transition: 'width .3s',
        }} />
      </div>
      <p style={{ fontSize: 11, color: 'var(--muted)', marginTop: 4 }}>{progress.msg}</p>
    </div>
  )
}
