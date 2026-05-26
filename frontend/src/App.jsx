import { useState, useRef, useCallback } from 'react'
import UploadPanel from './components/UploadPanel'
import VideoStream from './components/VideoStream'
import StatsBar from './components/StatsBar'
import ViolationPanel from './components/ViolationPanel'
import ProgressBar from './components/ProgressBar'
import Header from './components/Header'
import HistoryPanel from './components/HistoryPanel'
import './App.css'

const API = ''

/** Chuyển đường dẫn local (Windows) thành URL tĩnh /static/evidence/... */
const toEvidenceUrl = (p) => {
  if (!p) return null
  const normalized = p.replace(/\\/g, '/')
  const idx = normalized.indexOf('evidence/')
  return idx >= 0 ? `/static/${normalized.slice(idx)}` : null
}

export default function App() {
  const [activeTab, setActiveTab] = useState('monitor')
  const [sessionId, setSessionId] = useState(null)
  const [status, setStatus] = useState('idle')
  const [progress, setProgress] = useState({ pct: 0, msg: 'Chọn video để bắt đầu' })
  const [stats, setStats] = useState({ frame: 0, total: 0, light: 'unknown', violations: 0, fps: 0, stop_line_ok: null })
  const [violations, setViolations] = useState([])
  const [calibImg, setCalibImg] = useState(null)
  const [logs, setLogs] = useState([])
  const logsRef = useRef([])

  const wsRef = useRef(null)
  const canvasRef = useRef(null)
  const lastFrameTs = useRef(0)
  const fpsRef = useRef(0)

  const renderFrame = useCallback((arrayBuf) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const blob = new Blob([arrayBuf], { type: 'image/jpeg' })
    const url = URL.createObjectURL(blob)
    const img = new Image()
    img.onload = () => {
      canvas.width = img.naturalWidth
      canvas.height = img.naturalHeight
      canvas.getContext('2d').drawImage(img, 0, 0)
      URL.revokeObjectURL(url)
      const now = performance.now()
      if (lastFrameTs.current) fpsRef.current = Math.round(1000 / (now - lastFrameTs.current))
      lastFrameTs.current = now
    }
    img.src = url
  }, [])

  const connectWS = useCallback((sid) => {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${proto}://${location.host}/ws/${sid}`)
    ws.binaryType = 'arraybuffer'
    wsRef.current = ws

    ws.onopen = () => setStatus('calibrating')

    ws.onmessage = (ev) => {
      if (ev.data instanceof ArrayBuffer) {
        renderFrame(ev.data)
        setStats((prev) => ({ ...prev, fps: fpsRef.current }))
        return
      }

      const msg = JSON.parse(ev.data)
      if (msg.type === 'stats') {
        setStatus('processing')
        setStats({
          frame: msg.frame,
          total: msg.total,
          light: msg.light,
          violations: msg.violations,
          fps: fpsRef.current,
          stop_line_ok: msg.stop_line_ok,
        })
        setProgress({ pct: msg.progress, msg: `Frame ${msg.frame} / ${msg.total}` })
      } else if (msg.type === 'progress') {
        setProgress({ pct: msg.pct, msg: msg.msg })
        // Log khi phát hiện vạch dừng
        if (msg.msg && (msg.msg.includes('vạch dừng') || msg.msg.includes('Không tìm'))) {
          const ts = new Date().toLocaleTimeString('vi-VN')
          const isOk = msg.msg.includes('✅')
          const entry = { ts, type: isOk ? 'vach_ok' : 'vach_fail', text: msg.msg }
          logsRef.current = [entry, ...logsRef.current].slice(0, 200)
          setLogs([...logsRef.current])
        }
      } else if (msg.type === 'calib') {
        const blob = new Blob([new Uint8Array(msg.data)], { type: 'image/jpeg' })
        setCalibImg(URL.createObjectURL(blob))
      } else if (msg.type === 'violation') {
        const d = msg.data
        const enriched = {
          ...d,
          evidenceUrl: toEvidenceUrl(d.evidence),
          cropUrl: null,  // cropUrl sẽ cập nhật qua plate_update
        }
        setViolations((prev) => [enriched, ...prev].slice(0, 100))
        const ts = new Date().toLocaleTimeString('vi-VN')
        const entry = {
          ts, type: 'violation',
          text: `⚠️ [VI PHẠM] ${d.type} — ID${d.track_id} (${d.vehicle}) | t=${d.time}s${d.plate ? ` | BS: ${d.plate}` : ''}`
        }
        logsRef.current = [entry, ...logsRef.current].slice(0, 200)
        setLogs([...logsRef.current])
      } else if (msg.type === 'done') {
        setStatus('done')
        const s = msg.summary
        const ts = new Date().toLocaleTimeString('vi-VN')
        const entry = {
          ts, type: 'done',
          text: `✅ Hoàn tất! ${s.redlight_count} vượt đèn đỏ · ${s.helmet_count} không mũ · tổng ${s.total_frames} frame`
        }
        logsRef.current = [entry, ...logsRef.current].slice(0, 200)
        setLogs([...logsRef.current])
        setProgress({ pct: 1, msg: `Xong! ${s.redlight_count} vượt đèn đỏ · ${s.helmet_count} không mũ` })
      } else if (msg.type === 'plate_update') {
        const { track_id, plate, time, bbox, crop_path } = msg.data
        const cropUrl = toEvidenceUrl(crop_path)
        setViolations((prev) =>
          prev.map((v) =>
            v.track_id === track_id && !v.plate
              ? { ...v, plate, cropUrl: cropUrl || v.cropUrl }
              : v
          )
        )
        const ts = new Date().toLocaleTimeString('vi-VN')
        const bboxStr = bbox ? `bbox=[${bbox.join(',')}]` : ''
        const cropStr = crop_path ? ` | crop: ${crop_path.split(/[\\/]/).pop()}` : ''
        const entry = {
          ts, type: 'vach_ok',
          text: `🔢 [BIỂN SỐ CẬP NHẬT] ID${track_id}: [${plate}] | t=${time}s${bboxStr ? ' | ' + bboxStr : ''}${cropStr}`
        }
        logsRef.current = [entry, ...logsRef.current].slice(0, 200)
        setLogs([...logsRef.current])
      } else if (msg.type === 'error') {
        setStatus('idle')
        setProgress({ pct: 0, msg: `Lỗi: ${msg.msg}` })
      }
    }

    ws.onerror = () => {
      setProgress((prev) => ({ ...prev, msg: 'Lỗi WebSocket, vui lòng thử lại' }))
    }

    ws.onclose = () => {
      setStatus((prev) => (prev === 'done' ? prev : 'idle'))
    }
  }, [renderFrame])

  const handleStart = useCallback(async (file, cfg) => {
    setStatus('uploading')
    setViolations([])
    setCalibImg(null)
    setLogs([])
    logsRef.current = []
    setProgress({ pct: 0, msg: 'Đang upload video...' })

    const fd = new FormData()
    fd.append('file', file)
    Object.entries(cfg).forEach(([k, v]) => fd.append(k, v))

    try {
      const res = await fetch(`${API}/api/upload`, { method: 'POST', body: fd })
      const data = await res.json()
      setSessionId(data.session_id)
      connectWS(data.session_id)
    } catch (e) {
      setStatus('idle')
      setProgress({ pct: 0, msg: 'Lỗi upload: ' + e.message })
    }
  }, [connectWS])

  const handleStop = useCallback(() => {
    if (wsRef.current) wsRef.current.close()
    setStatus('idle')
  }, [])

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-slate-50">
      <Header status={status} activeTab={activeTab} setActiveTab={setActiveTab} />
      {activeTab === 'monitor' ? (
        <div className="flex flex-1 min-h-0 overflow-hidden">
          <UploadPanel onStart={handleStart} onStop={handleStop} status={status} />
          <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
            <ProgressBar progress={progress} />
            <VideoStream canvasRef={canvasRef} status={status} calibImg={calibImg} logs={logs} />
            <StatsBar stats={stats} />
          </div>
          <ViolationPanel violations={violations} />
        </div>
      ) : (
        <HistoryPanel />
      )}
    </div>
  )
}
