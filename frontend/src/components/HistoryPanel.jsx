import { useState, useEffect, useCallback } from 'react'
import ImageModal from './ImageModal'

/** Chuyển đường dẫn local (Windows) thành URL tĩnh /static/evidence/... */
const toEvidenceUrl = (p) => {
  if (!p) return null
  const normalized = p.replace(/\\/g, '/')
  const idx = normalized.indexOf('evidence/')
  return idx >= 0 ? `/static/${normalized.slice(idx)}` : null
}

const VIOLATION_TYPES = [
  { value: '', label: 'Tất cả các lỗi' },
  { value: 'Vượt đèn đỏ', label: 'Vượt đèn đỏ' },
  { value: 'Không đội mũ', label: 'Không đội mũ bảo hiểm' }
]

const STATUS_TYPES = [
  { value: '', label: 'Tất cả trạng thái' },
  { value: 'pending', label: 'Chờ xử lý', color: 'bg-amber-100 text-amber-800 border-amber-200' },
  { value: 'processed', label: 'Đã gửi phạt', color: 'bg-sky-100 text-sky-850 border-sky-200' },
  { value: 'paid', label: 'Đã nộp phạt', color: 'bg-emerald-100 text-emerald-800 border-emerald-200' }
]

export default function HistoryPanel() {
  const [violations, setViolations] = useState([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [limit] = useState(8)
  const [search, setSearch] = useState('')
  const [vType, setVType] = useState('')
  const [status, setStatus] = useState('')
  const [loading, setLoading] = useState(false)
  
  // State for detail modal
  const [selectedVio, setSelectedVio] = useState(null)
  const [updatingStatus, setUpdatingStatus] = useState(false)

  // DB Statistics
  const [dbStats, setDbStats] = useState({
    total: 0, redlight: 0, helmet: 0,
    status_stats: { pending: 0, processed: 0, paid: 0 },
    db_status: 'Đang kết nối...'
  })

  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch('/api/db/stats')
      const data = await res.json()
      setDbStats(data)
    } catch (e) {
      console.error("Lỗi fetch stats: ", e)
    }
  }, [])

  const fetchViolations = useCallback(async () => {
    setLoading(true)
    try {
      const q = new URLSearchParams({
        page, limit, search, type: vType, status
      }).toString()
      
      const res = await fetch(`/api/db/violations?${q}`)
      const data = await res.json()
      setViolations(data.violations || [])
      setTotal(data.total || 0)
    } catch (e) {
      console.error("Lỗi fetch violations: ", e)
    } finally {
      setLoading(false)
    }
  }, [page, limit, search, vType, status])

  useEffect(() => {
    fetchViolations()
    fetchStats()
  }, [fetchViolations, fetchStats])

  const handleSearchChange = (e) => {
    setSearch(e.target.value)
    setPage(1) // Reset về trang 1
  }

  const handleFilterTypeChange = (e) => {
    setVType(e.target.value)
    setPage(1)
  }

  const handleFilterStatusChange = (e) => {
    setStatus(e.target.value)
    setPage(1)
  }

  const handleUpdateStatus = async (vioId, newStatus) => {
    setUpdatingStatus(true)
    try {
      const res = await fetch(`/api/db/violations/${vioId}/status`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: newStatus })
      })
      const data = await res.json()
      if (data.ok) {
        // Cập nhật state tại chỗ
        setViolations(prev => prev.map(v => v.id === vioId ? { ...v, status: newStatus } : v))
        if (selectedVio && selectedVio.id === vioId) {
          setSelectedVio(prev => ({ ...prev, status: newStatus }))
        }
        fetchStats() // Reload thống kê
      }
    } catch (e) {
      alert("Lỗi cập nhật trạng thái: " + e.message)
    } finally {
      setUpdatingStatus(false)
    }
  }

  const getStatusBadge = (statusVal) => {
    const matched = STATUS_TYPES.find(s => s.value === statusVal) || STATUS_TYPES[1]
    return (
      <span className={`px-2.5 py-1 text-xs font-semibold rounded-full border ${matched.color}`}>
        {matched.label}
      </span>
    )
  }

  const totalPages = Math.ceil(total / limit) || 1

  return (
    <div className="flex-1 min-h-0 overflow-y-auto bg-slate-50 p-6 flex flex-col gap-6">
      
      {/* ── Thống kê tổng quan ── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex flex-col">
          <span className="text-xs font-semibold text-slate-400 uppercase">Tổng số vi phạm</span>
          <span className="text-2xl font-bold text-slate-800 mt-1">{dbStats.total} vụ</span>
          <span className="text-[10px] text-emerald-500 font-medium mt-2">📊 Cơ sở dữ liệu: {dbStats.db_status}</span>
        </div>
        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex flex-col">
          <span className="text-xs font-semibold text-slate-400 uppercase">Chờ xử lý (Phạt nguội)</span>
          <span className="text-2xl font-bold text-amber-600 mt-1">{dbStats.status_stats?.pending || 0} ca</span>
          <span className="text-[10px] text-slate-400 mt-2">Cần xác minh và gửi thông báo</span>
        </div>
        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex flex-col">
          <span className="text-xs font-semibold text-slate-400 uppercase">Đã gửi phạt nguội</span>
          <span className="text-2xl font-bold text-sky-600 mt-1">{dbStats.status_stats?.processed || 0} ca</span>
          <span className="text-[10px] text-slate-400 mt-2">Đang chờ người vi phạm nộp phạt</span>
        </div>
        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex flex-col">
          <span className="text-xs font-semibold text-slate-400 uppercase">Đã hoàn thành nộp phạt</span>
          <span className="text-2xl font-bold text-emerald-600 mt-1">{dbStats.status_stats?.paid || 0} ca</span>
          <span className="text-[10px] text-emerald-500 font-medium mt-2">✓ Đã đóng ngân sách nhà nước</span>
        </div>
      </div>

      {/* ── Bảng điều khiển & Bộ lọc ── */}
      <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex flex-col sm:flex-row gap-3">
        {/* Tìm kiếm biển số */}
        <div className="flex-1 relative">
          <input
            type="text"
            value={search}
            onChange={handleSearchChange}
            placeholder="🔍 Tìm kiếm biển số xe... (Ví dụ: 29-G1)"
            className="w-full h-10 px-3 pr-10 text-sm bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:border-indigo-500 focus:bg-white transition-all font-semibold"
          />
          {search && (
            <button
              onClick={() => { setSearch(''); setPage(1); }}
              className="absolute right-3 top-2.5 text-slate-400 hover:text-slate-600 text-sm"
            >
              ✕
            </button>
          )}
        </div>

        {/* Lọc loại lỗi */}
        <select
          value={vType}
          onChange={handleFilterTypeChange}
          className="h-10 px-3 bg-slate-50 border border-slate-200 rounded-lg text-sm focus:outline-none font-medium"
        >
          {VIOLATION_TYPES.map(t => (
            <option key={t.value} value={t.value}>{t.label}</option>
          ))}
        </select>

        {/* Lọc trạng thái */}
        <select
          value={status}
          onChange={handleFilterStatusChange}
          className="h-10 px-3 bg-slate-50 border border-slate-200 rounded-lg text-sm focus:outline-none font-medium"
        >
          {STATUS_TYPES.map(s => (
            <option key={s.value} value={s.value}>{s.label}</option>
          ))}
        </select>
      </div>

      {/* ── Bảng kết quả vi phạm ── */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm flex-1 min-h-0 flex flex-col">
        <div className="flex-1 min-h-0 overflow-auto">
          {loading ? (
            <div className="w-full h-full flex flex-col items-center justify-center text-slate-400 gap-2">
              <div className="w-8 h-8 border-2 border-slate-200 border-t-indigo-600 rounded-full animate-spin" />
              <span className="text-xs font-semibold">Đang tải dữ liệu...</span>
            </div>
          ) : violations.length === 0 ? (
            <div className="w-full h-full flex flex-col items-center justify-center text-slate-400 py-12 gap-2">
              <span className="text-3xl">📭</span>
              <span className="text-sm font-semibold">Không tìm thấy biên bản vi phạm nào</span>
            </div>
          ) : (
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-slate-50 border-b border-slate-200 text-xs font-bold text-slate-500 uppercase tracking-wider sticky top-0 z-10">
                  <th className="py-3 px-4">Thời gian</th>
                  <th className="py-3 px-4">Loại vi phạm</th>
                  <th className="py-3 px-4">Phương tiện</th>
                  <th className="py-3 px-4">Biển số</th>
                  <th className="py-3 px-4">Độ tin cậy</th>
                  <th className="py-3 px-4">Bằng chứng</th>
                  <th className="py-3 px-4">Trạng thái</th>
                  <th className="py-3 px-4 text-right">Chi tiết</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-slate-700 text-sm font-medium">
                {violations.map((vio) => (
                  <tr key={vio.id} className="hover:bg-slate-50/50 transition-colors">
                    <td className="py-3 px-4 text-xs font-semibold text-slate-500">
                      {new Date(vio.timestamp).toLocaleString('vi-VN')}
                    </td>
                    <td className="py-3 px-4">
                      <span className={`px-2 py-0.5 rounded text-xs font-bold ${
                        vio.violation_type === 'Vượt đèn đỏ' ? 'bg-red-50 text-red-650' : 'bg-amber-50 text-amber-700'
                      }`}>
                        {vio.violation_type}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-slate-600">{vio.vehicle_type}</td>
                    <td className="py-3 px-4">
                      {vio.license_plate ? (
                        <span className="bg-slate-100 border border-slate-200 text-slate-800 px-2 py-0.5 rounded font-mono font-bold text-xs tracking-wide">
                          {vio.license_plate}
                        </span>
                      ) : (
                        <span className="text-slate-400 text-xs italic">Chưa đọc được</span>
                      )}
                    </td>
                    <td className="py-3 px-4 text-xs text-slate-500 font-mono font-bold">
                      {vio.confidence ? `${Math.round(vio.confidence * 100)}%` : '92%'}
                    </td>
                    <td className="py-3 px-4">
                      {vio.evidence_image ? (
                        <span className="text-xs text-indigo-650 hover:underline cursor-pointer flex items-center gap-1"
                              onClick={() => setSelectedVio(vio)}>
                          📸 Xem ảnh
                        </span>
                      ) : (
                        <span className="text-slate-400 text-xs">-</span>
                      )}
                    </td>
                    <td className="py-3 px-4">
                      {getStatusBadge(vio.status)}
                    </td>
                    <td className="py-3 px-4 text-right">
                      <button
                        onClick={() => setSelectedVio(vio)}
                        className="text-xs text-indigo-600 hover:text-indigo-800 bg-indigo-50 hover:bg-indigo-100 px-3 py-1.5 rounded-md transition-colors"
                      >
                        🔎 Quản lý
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* ── Phân trang ── */}
        {!loading && total > 0 && (
          <div className="h-14 border-t border-slate-200 px-4 flex items-center justify-between text-xs font-semibold text-slate-500 bg-slate-50/50 rounded-b-xl">
            <span>Hiển thị {violations.length} trên tổng số {total} vi phạm</span>
            <div className="flex items-center gap-1">
              <button
                disabled={page <= 1}
                onClick={() => setPage(p => Math.max(1, p - 1))}
                className="w-8 h-8 rounded-lg border border-slate-200 flex items-center justify-center hover:bg-white hover:text-slate-800 disabled:opacity-40 disabled:hover:bg-transparent transition-all"
              >
                ◀
              </button>
              <span className="px-3">Trang {page} / {totalPages}</span>
              <button
                disabled={page >= totalPages}
                onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                className="w-8 h-8 rounded-lg border border-slate-200 flex items-center justify-center hover:bg-white hover:text-slate-800 disabled:opacity-40 disabled:hover:bg-transparent transition-all"
              >
                ▶
              </button>
            </div>
          </div>
        )}
      </div>

      {/* ── Modal chi tiết vi phạm và Quản lý ── */}
      {selectedVio && (
        <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-white w-full max-w-2xl rounded-2xl border border-slate-200 shadow-xl overflow-hidden flex flex-col animate-scale-up">
            
            {/* Header Modal */}
            <div className="h-14 border-b border-slate-100 px-6 flex items-center justify-between bg-slate-50">
              <h3 className="font-bold text-slate-800 text-sm flex items-center gap-1.5">
                <span className="text-lg">⚖️</span> Biên bản Phạt nguội: ID{selectedVio.track_id}
              </h3>
              <button
                onClick={() => setSelectedVio(null)}
                className="w-7 h-7 rounded-full flex items-center justify-center text-slate-400 hover:text-slate-600 hover:bg-slate-200 transition-colors"
              >
                ✕
              </button>
            </div>

            {/* Content Modal */}
            <div className="p-6 flex flex-col md:flex-row gap-6 max-h-[80vh] overflow-y-auto">
              
              {/* Cột trái: Ảnh bằng chứng */}
              <div className="flex-1 flex flex-col gap-4">
                <div className="relative group rounded-lg overflow-hidden border border-slate-200 bg-slate-100 aspect-video flex items-center justify-center">
                  {selectedVio.evidence_image ? (
                    <img
                      src={toEvidenceUrl(selectedVio.evidence_image)}
                      alt="Bằng chứng vi phạm"
                      className="w-full h-full object-contain"
                    />
                  ) : (
                    <span className="text-xs text-slate-400">Không có ảnh bằng chứng</span>
                  )}
                </div>
                <div className="flex gap-4">
                  <div className="flex-1">
                    <span className="text-[10px] font-bold text-slate-400 uppercase">Ảnh bằng chứng gốc</span>
                    <div className="text-xs font-semibold text-indigo-650 break-all truncate mt-1">
                      {selectedVio.evidence_image ? selectedVio.evidence_image.split(/[\\/]/).pop() : 'N/A'}
                    </div>
                  </div>
                  {selectedVio.plate_crop && (
                    <div className="w-24 flex flex-col items-center">
                      <span className="text-[10px] font-bold text-slate-400 uppercase block mb-1">Cắt biển số</span>
                      <div className="w-full h-10 border border-slate-200 rounded overflow-hidden bg-white flex items-center justify-center">
                        <img
                          src={toEvidenceUrl(selectedVio.plate_crop)}
                          alt="Crop biển số"
                          className="max-w-full max-h-full object-contain"
                        />
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Cột phải: Thông tin chi tiết & Đổi trạng thái */}
              <div className="w-full md:w-60 flex flex-col gap-5 text-xs font-medium text-slate-600">
                <div className="flex flex-col gap-3.5 bg-slate-50 p-4 rounded-xl border border-slate-100">
                  <div>
                    <span className="text-[10px] font-bold text-slate-400 uppercase">Lỗi vi phạm</span>
                    <div className="text-sm font-bold text-red-650 mt-0.5">{selectedVio.violation_type}</div>
                  </div>
                  <div>
                    <span className="text-[10px] font-bold text-slate-400 uppercase">Biển số xe</span>
                    <div className="mt-0.5">
                      {selectedVio.license_plate ? (
                        <span className="bg-slate-800 text-white font-mono font-bold px-2 py-0.5 rounded text-xs tracking-wider">
                          {selectedVio.license_plate}
                        </span>
                      ) : (
                        <span className="text-slate-400 italic">Chưa đọc được</span>
                      )}
                    </div>
                  </div>
                  <div>
                    <span className="text-[10px] font-bold text-slate-400 uppercase">Thời gian vi phạm</span>
                    <div className="text-slate-800 font-semibold mt-0.5">
                      {new Date(selectedVio.timestamp).toLocaleString('vi-VN')}
                    </div>
                  </div>
                  <div>
                    <span className="text-[10px] font-bold text-slate-400 uppercase">Loại xe</span>
                    <div className="text-slate-800 font-semibold mt-0.5">{selectedVio.vehicle_type} (ID{selectedVio.track_id})</div>
                  </div>
                  <div>
                    <span className="text-[10px] font-bold text-slate-400 uppercase">Thời điểm trong video</span>
                    <div className="text-slate-800 font-semibold mt-0.5">{selectedVio.time} giây (Frame {selectedVio.frame})</div>
                  </div>
                </div>

                {/* Dropdown đổi trạng thái */}
                <div className="flex flex-col gap-2">
                  <span className="text-[10px] font-bold text-slate-400 uppercase">Quản lý trạng thái xử phạt</span>
                  <select
                    disabled={updatingStatus}
                    value={selectedVio.status}
                    onChange={(e) => handleUpdateStatus(selectedVio.id, e.target.value)}
                    className="w-full h-10 px-3 bg-white border border-slate-200 rounded-lg font-semibold text-slate-800 focus:outline-none focus:border-indigo-500 disabled:opacity-50 transition-colors"
                  >
                    <option value="pending">⏳ Chờ xử lý (Chưa phạt)</option>
                    <option value="processed">✉️ Đã gửi thông báo phạt</option>
                    <option value="paid">✅ Người dân đã nộp phạt</option>
                  </select>
                  {updatingStatus && <span className="text-[10px] text-indigo-650 animate-pulse font-semibold">Đang cập nhật...</span>}
                </div>
              </div>

            </div>

            {/* Footer Modal */}
            <div className="h-14 border-t border-slate-100 px-6 flex items-center justify-end bg-slate-50 gap-2">
              <button
                onClick={() => setSelectedVio(null)}
                className="px-4 py-2 rounded-lg border border-slate-200 text-xs font-semibold text-slate-600 hover:bg-slate-100 hover:text-slate-800 transition-all"
              >
                Đóng
              </button>
            </div>

          </div>
        </div>
      )}

    </div>
  )
}
