import { useRef, useState } from 'react'

const DEFAULT_CFG = {
    conf_light: 0.5,
    conf_vehicle: 0.4,
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

function RangeField({ label, value, min, max, step, onChange, help, formatValue }) {
    return (
        <div className="field">
            <label>
                <span>{label}</span>
                <span className="val">{formatValue ? formatValue(value) : value}</span>
            </label>
            <input type="range" min={min} max={max} step={step} value={value} onChange={onChange} />
            {help ? <p className="field-help">{help}</p> : null}
        </div>
    )
}

function SelectField({ label, value, onChange, help, children }) {
    return (
        <div className="field">
            <label>{label}</label>
            <select value={value} onChange={onChange}>
                {children}
            </select>
            {help ? <p className="field-help">{help}</p> : null}
        </div>
    )
}

function PanelSection({ title, caption, children }) {
    return (
        <>
            <p className="section-title">{title}</p>
            <div className="card">
                {caption ? <p className="card-caption">{caption}</p> : null}
                {children}
            </div>
        </>
    )
}

export default function UploadPanel({ onStart, onStop, status }) {
    const [file, setFile] = useState(null)
    const [cfg, setCfg] = useState(DEFAULT_CFG)
    const [dragging, setDrag] = useState(false)
    const inputRef = useRef(null)

    const busy = status !== 'idle' && status !== 'done'

    const setNum = (k, v) => setCfg((prev) => ({ ...prev, [k]: Number(v) }))
    const setBool = (k, v) => setCfg((prev) => ({ ...prev, [k]: v }))
    const pick = (f) => {
        if (f) setFile(f)
    }

    return (
        <aside className="upload-panel">
            <div className="upload-panel-scroll">
                <p className="section-title">Video</p>
                <div
                    className={`dropzone${dragging ? ' is-dragging' : ''}`}
                    onClick={() => inputRef.current?.click()}
                    onDragOver={(e) => {
                        e.preventDefault()
                        setDrag(true)
                    }}
                    onDragLeave={() => setDrag(false)}
                    onDrop={(e) => {
                        e.preventDefault()
                        setDrag(false)
                        pick(e.dataTransfer.files?.[0])
                    }}
                >
                    <div className="dropzone-icon">🎬</div>
                    <strong>Tải video giao thông</strong>
                    <p>Kéo thả hoặc bấm để chọn file `.mp4`, `.avi`, `.mov`</p>
                    {file ? <div className="file-pill">✅ {file.name}</div> : null}
                </div>
                <input
                    ref={inputRef}
                    type="file"
                    accept=".mp4,.avi,.mov"
                    style={{ display: 'none' }}
                    onChange={(e) => pick(e.target.files?.[0])}
                />

                <PanelSection title="⚙️ Cài đặt" caption="Model đã được khóa cứng, phần này chỉ chỉnh ngưỡng detect.">
                    <RangeField
                        label="Confidence: Đèn"
                        value={cfg.conf_light}
                        min="0.1"
                        max="0.9"
                        step="0.05"
                        onChange={(e) => setNum('conf_light', e.target.value)}
                        formatValue={(v) => v.toFixed(2)}
                    />
                    <RangeField
                        label="Confidence: Xe"
                        value={cfg.conf_vehicle}
                        min="0.1"
                        max="0.9"
                        step="0.05"
                        onChange={(e) => setNum('conf_vehicle', e.target.value)}
                        formatValue={(v) => v.toFixed(2)}
                    />
                    <RangeField
                        label="Confidence: Mũ"
                        value={cfg.conf_helmet}
                        min="0.1"
                        max="0.9"
                        step="0.05"
                        onChange={(e) => setNum('conf_helmet', e.target.value)}
                        formatValue={(v) => v.toFixed(2)}
                    />
                </PanelSection>

                <PanelSection title="🚧 Cài đặt Vạch Dừng (SAM)" caption="Model vạch dừng đã khóa cứng, bạn chỉ chỉnh tham số calibration.">
                    <RangeField
                        label="Confidence: Vạch dừng"
                        value={cfg.conf_stop_line}
                        min="0.1"
                        max="0.9"
                        step="0.05"
                        onChange={(e) => setNum('conf_stop_line', e.target.value)}
                        help="Ngưỡng tin cậy cho model detect vạch dừng."
                        formatValue={(v) => v.toFixed(2)}
                    />
                    <RangeField
                        label="Min width vạch (% frame)"
                        value={cfg.min_width_pct}
                        min="1"
                        max="40"
                        step="1"
                        onChange={(e) => setNum('min_width_pct', e.target.value)}
                        help="Video lớn có thể giảm xuống 5-10%, mặc định 8%."
                    />
                    <RangeField
                        label="Mở rộng trái (px)"
                        value={cfg.stop_line_extend_left}
                        min="0"
                        max="500"
                        step="10"
                        onChange={(e) => setNum('stop_line_extend_left', e.target.value)}
                        help="Kéo dài vạch sang trái để phủ toàn bộ làn đường."
                    />
                    <RangeField
                        label="Mở rộng phải (px)"
                        value={cfg.stop_line_extend_right}
                        min="0"
                        max="500"
                        step="10"
                        onChange={(e) => setNum('stop_line_extend_right', e.target.value)}
                        help="Kéo dài vạch sang phải để phủ toàn bộ làn đường."
                    />
                    <RangeField
                        label="Dịch vạch lên (px)"
                        value={cfg.stop_line_offset_up}
                        min="0"
                        max="100"
                        step="5"
                        onChange={(e) => setNum('stop_line_offset_up', e.target.value)}
                        help="Giảm false-positive bằng cách nâng ngưỡng vi phạm lên cao hơn."
                    />
                </PanelSection>

                <PanelSection title="🔍 Tính năng" caption="Bật các pipeline vi phạm cần dùng.">
                    <label className="toggle">
                        <input
                            type="checkbox"
                            checked={cfg.check_redlight}
                            onChange={(e) => setBool('check_redlight', e.target.checked)}
                        />
                        <span>🔴 Phát hiện vượt đèn đỏ</span>
                    </label>
                    <label className="toggle">
                        <input
                            type="checkbox"
                            checked={cfg.check_helmet}
                            onChange={(e) => setBool('check_helmet', e.target.checked)}
                        />
                        <span>🪖 Kiểm tra mũ bảo hiểm</span>
                    </label>
                    <label className="toggle">
                        <input
                            type="checkbox"
                            checked={cfg.check_plate}
                            onChange={(e) => setBool('check_plate', e.target.checked)}
                        />
                        <span>🔢 Nhận dạng biển số (OCR)</span>
                    </label>

                    {cfg.check_plate ? (
                        <div className="inline-subsection">
                            <p className="subsection-caption">Model biển số đã khóa cứng sang `biensoxe.onnx`, bạn chỉ chỉnh độ nhạy nhận diện.</p>
                            <RangeField
                                label="YOLO Biển số - conf"
                                value={cfg.conf_lp}
                                min="0.05"
                                max="0.5"
                                step="0.05"
                                onChange={(e) => setNum('conf_lp', e.target.value)}
                                help="Giảm nếu biển số nhỏ hoặc dễ bị bỏ sót."
                                formatValue={(v) => v.toFixed(2)}
                            />
                            <RangeField
                                label="OCR Score - ngưỡng chấp nhận"
                                value={cfg.conf_ocr}
                                min="0.05"
                                max="0.9"
                                step="0.05"
                                onChange={(e) => setNum('conf_ocr', e.target.value)}
                                formatValue={(v) => v.toFixed(2)}
                            />
                            <SelectField
                                label="LP YOLO imgsz"
                                value={cfg.lp_imgsz}
                                onChange={(e) => setNum('lp_imgsz', e.target.value)}
                                help="640 là cân bằng, 1024 ưu tiên cho ONNX nặng."
                            >
                                <option value={320}>320 - rất nhanh</option>
                                <option value={640}>640 - cân bằng</option>
                                <option value={1024}>1024 - chính xác hơn</option>
                            </SelectField>
                        </div>
                    ) : null}
                </PanelSection>

                <PanelSection title="⚡ Hiệu năng" caption="Giữ tinh thần của sidebar cũ: giảm N và imgsz nếu cần tốc độ.">
                    <RangeField
                        label="Xử lý mỗi N frame (AI)"
                        value={cfg.process_every_n}
                        min="1"
                        max="5"
                        step="1"
                        onChange={(e) => setNum('process_every_n', e.target.value)}
                    />
                    <RangeField
                        label="Check đèn mỗi N frame"
                        value={cfg.traffic_interval}
                        min="1"
                        max="10"
                        step="1"
                        onChange={(e) => setNum('traffic_interval', e.target.value)}
                    />
                    <RangeField
                        label="Cập nhật UI mỗi N frame"
                        value={cfg.display_every_n}
                        min="1"
                        max="10"
                        step="1"
                        onChange={(e) => setNum('display_every_n', e.target.value)}
                        help="Tăng lên nếu muốn demo mượt hơn trên máy yếu."
                    />
                    <SelectField
                        label="Độ rộng hiển thị (px)"
                        value={cfg.display_width}
                        onChange={(e) => setNum('display_width', e.target.value)}
                        help="Resize trước khi gửi lên UI để đỡ lag, giống cách Streamlit cũ tối ưu preview."
                    >
                        <option value={320}>320</option>
                        <option value={480}>480</option>
                        <option value={640}>640</option>
                        <option value={800}>800</option>
                        <option value={960}>960</option>
                        <option value={1280}>1280</option>
                    </SelectField>
                    <SelectField
                        label="YOLO imgsz (xe/đèn)"
                        value={cfg.yolo_imgsz}
                        onChange={(e) => setNum('yolo_imgsz', e.target.value)}
                        help="640 là mặc định cũ, 480 nhanh hơn cho máy yếu."
                    >
                        <option value={320}>320</option>
                        <option value={480}>480</option>
                        <option value={640}>640</option>
                        <option value={960}>960</option>
                        <option value={1280}>1280</option>
                    </SelectField>
                    <label className="toggle">
                        <input
                            type="checkbox"
                            checked={cfg.show_all}
                            onChange={(e) => setBool('show_all', e.target.checked)}
                        />
                        <span>👁️ Hiện tất cả xe</span>
                    </label>
                </PanelSection>
            </div>

            <div className="panel-actions">
                <button className="btn btn-primary" disabled={!file || busy} onClick={() => onStart(file, cfg)}>
                    {busy ? 'Đang xử lý...' : 'Bắt đầu quét'}
                </button>
                <button className="btn btn-danger" disabled={!busy} onClick={onStop}>
                    Dừng
                </button>
            </div>
        </aside>
    )
}
