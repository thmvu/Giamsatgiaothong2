"""
FastAPI Backend — AI Traffic Monitor
=====================================
Chạy: uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
Frontend React (Vite): http://localhost:3000  ← mở cái này
"""

import asyncio, os, uuid, tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.processor import VideoProcessor, ProcessorConfig

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="AI Traffic Monitor", version="2.0")

# Serve ảnh bằng chứng tĩnh — tạo thư mục trước để tránh lỗi khi chưa có file nào
os.makedirs("evidence", exist_ok=True)
os.makedirs(os.path.join("evidence", "plates"), exist_ok=True)
app.mount("/static/evidence", StaticFiles(directory="evidence"), name="evidence")
executor = ThreadPoolExecutor(max_workers=2)

# CORS: cho phép React (port 3000) gọi FastAPI (port 8000) trong dev mode
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store  {session_id: {...}}
sessions: Dict[str, Dict[str, Any]] = {}

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "sessions": len(sessions)}

@app.post("/api/upload")
async def upload_video(
    file: UploadFile = File(...),
    conf_light: float = Form(0.5),
    conf_vehicle: float = Form(0.4),
    conf_helmet: float = Form(0.4),
    conf_lp: float = Form(0.2),
    conf_ocr: float = Form(0.5),
    conf_stop_line: float = Form(0.3),
    process_every_n: int = Form(2),
    traffic_interval: int = Form(5),
    display_every_n: int = Form(3),
    display_width: int = Form(640),
    yolo_imgsz: int = Form(640),
    lp_imgsz: int = Form(640),
    stop_line_extend_left: int = Form(150),
    stop_line_extend_right: int = Form(150),
    stop_line_offset_up: int = Form(30),
    min_width_pct: float = Form(8.0),
    check_redlight: bool = Form(True),
    check_helmet: bool = Form(True),
    check_plate: bool = Form(True),
    show_all: bool = Form(True),
):
    # Lưu video vào temp file
    suffix = os.path.splitext(file.filename)[-1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    content = await file.read()
    tmp.write(content)
    tmp.flush()
    tmp.close()

    session_id = uuid.uuid4().hex[:8]
    cfg = ProcessorConfig(
        conf_light=conf_light, conf_vehicle=conf_vehicle,
        conf_helmet=conf_helmet, conf_lp=conf_lp, conf_ocr=conf_ocr,
        conf_stop_line=conf_stop_line, process_every_n=process_every_n,
        traffic_interval=traffic_interval, display_every_n=display_every_n,
        display_width=display_width, yolo_imgsz=yolo_imgsz,
        lp_imgsz=lp_imgsz, stop_line_extend_left=stop_line_extend_left,
        stop_line_extend_right=stop_line_extend_right,
        stop_line_offset_up=stop_line_offset_up, min_width_pct=min_width_pct,
        check_redlight=check_redlight, check_helmet=check_helmet,
        check_plate=check_plate, show_all=show_all,
    )
    sessions[session_id] = {
        "video_path": tmp.name,
        "filename": file.filename,
        "cfg": cfg,
        "status": "ready",
        "progress": 0.0,
        "progress_msg": "Chờ bắt đầu...",
        "violations": [],
        "summary": None,
        "calib_frame": None,   # base64 JPEG of calibration result
    }
    return {"session_id": session_id, "filename": file.filename}


@app.websocket("/ws/{session_id}")
async def websocket_stream(ws: WebSocket, session_id: str):
    await ws.accept()
    session = sessions.get(session_id)
    if not session:
        await ws.close(code=1008, reason="Session not found")
        return

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=8)

    # Tạo processor và load model trong executor (KHÔNG block event loop)
    proc = VideoProcessor(session["cfg"])
    try:
        await loop.run_in_executor(executor, proc.load_models)
    except Exception as e:
        await ws.send_json({"type": "error", "msg": f"Lỗi load model: {e}"})
        await ws.close()
        return

    def _safe_run(coro):
        """Gửi coroutine lên asyncio event loop từ thread nền.
        Nếu loop đã đóng hoặc WebSocket đã ngắt → đóng coroutine tường minh
        để tránh RuntimeWarning 'coroutine was never awaited'.
        """
        if loop.is_closed():
            coro.close()
            return
        try:
            fut = asyncio.run_coroutine_threadsafe(coro, loop)
            fut.result(timeout=3)
        except Exception:
            # Đóng coroutine nếu chưa được consume để tránh warning
            try:
                coro.close()
            except Exception:
                pass

    def _put(frame_bytes, stats):
        """Thread-safe put into asyncio queue. Drop frame if queue full."""
        if loop.is_closed():
            return
        coro = _safe_put(queue, frame_bytes, stats)
        try:
            fut = asyncio.run_coroutine_threadsafe(coro, loop)
            fut.result(timeout=3)
        except Exception:
            try:
                coro.close()
            except Exception:
                pass

    def _on_calib(calib_bytes):
        session["calib_frame"] = calib_bytes
        _safe_run(ws.send_json({"type": "calib", "data": list(calib_bytes)}))

    def _on_violation(vio):
        session["violations"].append(vio)
        _safe_run(ws.send_json({"type": "violation", "data": vio}))

    def _on_progress(pct, msg):
        session["progress"] = pct
        session["progress_msg"] = msg
        _safe_run(ws.send_json({"type": "progress", "pct": pct, "msg": msg}))

    def _on_done(summary):
        session["status"] = "done"
        session["summary"] = summary
        _safe_run(ws.send_json({"type": "done", "summary": summary}))

    def _on_plate_update(info):
        """Gửi cập nhật biển số muộn cho frontend."""
        # Cập nhật trong session violations nếu có
        for v in session["violations"]:
            if v["track_id"] == info["track_id"] and not v["plate"]:
                v["plate"] = info["plate"]
        _safe_run(ws.send_json({"type": "plate_update", "data": info}))

    # Run AI in background thread
    future = executor.submit(
        proc.process,
        session["video_path"],
        _put,
        _on_violation,
        _on_calib,
        _on_done,
        _on_progress,
        _on_plate_update,
    )
    session["status"] = "processing"

    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=5)
            except asyncio.TimeoutError:
                if future.done():
                    exc = future.exception()
                    if exc is not None:
                        print(f"[PROCESS Error] {exc}")
                        await ws.send_json({"type": "error", "msg": f"Loi xu ly video: {exc}"})
                    break
                continue
            if item is None:
                break
            frame_bytes, stats = item
            # Send JPEG frame as binary
            await ws.send_bytes(frame_bytes)
            # Send stats as JSON text
            await ws.send_json({"type": "stats", **stats})
    except WebSocketDisconnect:
        proc.stop()
    except Exception as e:
        print(f"[WS Error] {e}")
    finally:
        proc.stop()
        future.cancel()


async def _safe_put(q: asyncio.Queue, frame_bytes, stats):
    if q.full():
        try: q.get_nowait()
        except asyncio.QueueEmpty: pass
    await q.put((frame_bytes, stats))


@app.get("/api/results/{session_id}")
async def get_results(session_id: str):
    session = sessions.get(session_id)
    if not session:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return {
        "status": session["status"],
        "violations": session["violations"],
        "summary": session.get("summary"),
    }


@app.get("/api/sessions")
async def list_sessions():
    return [{"id": sid, "filename": s["filename"], "status": s["status"]}
            for sid, s in sessions.items()]


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    session = sessions.pop(session_id, None)
    if session:
        try: os.unlink(session["video_path"])
        except: pass
    return {"ok": True}
