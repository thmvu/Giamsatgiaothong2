"""
Database Module — ITS Pro v2.0
===============================
Hỗ trợ kết nối bất đồng bộ tới MongoDB Atlas thông qua `motor`.
Có cơ chế Graceful Fallback tự động chuyển sang lưu trữ tạm thời (In-Memory)
nếu không thể kết nối tới MongoDB, đảm bảo ứng dụng không bao giờ bị crash.
"""

import os
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv

# Nạp các biến môi trường từ file .env
load_dotenv()

MONGODB_URL = os.getenv("MONGODB_URL")
DB_NAME = "its_traffic"
COLLECTION_NAME = "violations"

# Thiết lập Logger để dễ dàng debug
logger = logging.getLogger("its_database")
logging.basicConfig(level=logging.INFO)

# Trạng thái sẵn sàng của Database
MONGO_ENABLED = False
db_client = None
db = None
collection = None

# Danh sách in-memory dùng làm Fallback khi không có MongoDB
_fallback_violations: List[Dict[str, Any]] = []

def init_db():
    """
    Khởi tạo kết nối tới MongoDB Atlas.
    Hàm này được gọi khi FastAPI khởi động (startup event).
    """
    global MONGO_ENABLED, db_client, db, collection
    
    if not MONGODB_URL:
        logger.warning("⚠️  [DATABASE] MONGODB_URL trống! Chuyển sang chế độ In-Memory Fallback.")
        MONGO_ENABLED = False
        return

    try:
        from motor.motor_asyncio import AsyncIOMotorClient
        from pymongo.errors import ConnectionFailure
        
        logger.info(f"🔌 [DATABASE] Đang kết nối tới MongoDB Atlas...")
        db_client = AsyncIOMotorClient(MONGODB_URL, serverSelectionTimeoutMS=5000)
        db = db_client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        # Test kết nối bằng cách ping server (timeout 5s)
        # Vì motor là async, ta cần chạy trong event loop hiện tại hoặc tạo mới
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        if loop.is_running():
            # Nếu loop đang chạy (như trong FastAPI), ta tạo task để ping sau
            asyncio.create_task(_ping_db())
        else:
            loop.run_until_complete(_ping_db())
            
    except Exception as e:
        logger.error(f"❌ [DATABASE] Lỗi kết nối MongoDB: {e}. Chuyển sang In-Memory Fallback.")
        MONGO_ENABLED = False

async def _ping_db():
    global MONGO_ENABLED, db_client
    if db_client is None:
        MONGO_ENABLED = False
        return
    try:
        # Ping thử db_client để kích hoạt kết nối thực sự
        await db_client.admin.command('ping')
        MONGO_ENABLED = True
        logger.info("✅ [DATABASE] Kết nối MongoDB Atlas THÀNH CÔNG!")
    except Exception as e:
        logger.error(f"❌ [DATABASE] Ping MongoDB Atlas thất bại: {e}. Chuyển sang In-Memory Fallback.")
        MONGO_ENABLED = False


# ── Nghiệp vụ lưu và cập nhật vi phạm ───────────────────────────────────────

async def save_violation(vio_data: Dict[str, Any]) -> str:
    """
    Lưu mới một biên bản vi phạm vào MongoDB (hoặc in-memory fallback).
    
    Args:
        vio_data: Dữ liệu vi phạm gồm track_id, type, vehicle, time, frame, evidence...
    Returns:
        ID của bản ghi vi phạm vừa tạo (dưới dạng string).
    """
    import uuid
    
    # Chuẩn hóa dữ liệu lưu trữ theo đúng Schema báo cáo của người dùng
    violation_id = uuid.uuid4().hex[:12]
    document = {
        "_id": violation_id,
        "track_id": vio_data.get("track_id"),
        "timestamp": datetime.now().isoformat(),
        "violation_type": vio_data.get("type", "Không rõ"),
        "vehicle_type": vio_data.get("vehicle", "Xe máy"),
        "license_plate": vio_data.get("plate", ""),
        "confidence": vio_data.get("confidence", 0.92),  # Mức độ tin cậy AI mặc định
        "evidence_image": vio_data.get("evidence", ""),
        "plate_crop": vio_data.get("crop_path", ""),
        "status": "pending",  # pending (Chờ xử lý), processed (Đã gửi thông báo), paid (Đã nộp phạt)
        
        # Lưu thêm thông tin kỹ thuật video phục vụ đối chiếu
        "time": vio_data.get("time", 0.0),
        "frame": vio_data.get("frame", 0),
    }
    
    if MONGO_ENABLED and collection is not None:
        try:
            await collection.insert_one(document)
            logger.info(f"💾 [DATABASE] Đã lưu vi phạm {violation_id} (ID{document['track_id']}) vào MongoDB Atlas Cloud.")
            return violation_id
        except Exception as e:
            logger.error(f"❌ [DATABASE] Lỗi lưu MongoDB: {e}. Chuyển sang lưu In-Memory.")
            
    # Fallback ghi vào RAM
    _fallback_violations.insert(0, document)
    logger.info(f"💾 [DATABASE] [FALLBACK] Đã lưu vi phạm {violation_id} (ID{document['track_id']}) vào RAM.")
    return violation_id


async def update_violation_plate(track_id: int, plate_text: str, crop_path: Optional[str] = None, bbox: Optional[list] = None) -> bool:
    """
    Cập nhật biển số xe giải mã muộn (Late Update) cho các vi phạm của cùng một phương tiện.
    
    Args:
        track_id: ID theo vết của xe
        plate_text: Văn bản biển số đọc được
        crop_path: Đường dẫn ảnh cắt biển số xe
        bbox: Tọa độ biển số trong frame
    """
    if not track_id or not plate_text:
        return False
        
    update_fields: Dict[str, Any] = {"license_plate": plate_text}
    if crop_path:
        update_fields["plate_crop"] = crop_path
    if bbox:
        update_fields["plate_bbox"] = bbox
        
    if MONGO_ENABLED and collection is not None:
        try:
            # Cập nhật tất cả vi phạm có cùng track_id nhưng chưa có biển số
            res = await collection.update_many(
                {"track_id": track_id, "license_plate": ""},
                {"$set": update_fields}
            )
            logger.info(f"🔢 [DATABASE] Đã cập nhật biển số [{plate_text}] cho {res.modified_count} bản ghi MongoDB.")
            return res.modified_count > 0
        except Exception as e:
            logger.error(f"❌ [DATABASE] Lỗi cập nhật biển số MongoDB: {e}.")
            
    # Fallback cập nhật trong RAM
    updated = False
    for v in _fallback_violations:
        if v["track_id"] == track_id and not v["license_plate"]:
            v["license_plate"] = plate_text
            if crop_path:
                v["plate_crop"] = crop_path
            if bbox:
                v["plate_bbox"] = bbox
            updated = True
            
    if updated:
        logger.info(f"🔢 [DATABASE] [FALLBACK] Đã cập nhật biển số [{plate_text}] cho xe ID{track_id} trong RAM.")
    return updated


# ── Nghiệp vụ tra cứu và thống kê ───────────────────────────────────────────

async def get_violations(page: int = 1, limit: int = 10, search: str = "", 
                         v_type: str = "", status: str = "") -> Tuple[List[Dict[str, Any]], int]:
    """
    Lấy danh sách các vụ vi phạm hỗ trợ phân trang, tìm kiếm và bộ lọc.
    
    Returns:
        Tuple: (danh sách bản ghi, tổng số bản ghi thỏa mãn bộ lọc)
    """
    skip = (page - 1) * limit
    
    if MONGO_ENABLED and collection is not None:
        try:
            # Tạo bộ lọc truy vấn
            query = {}
            if search:
                # Tìm kiếm regex không phân biệt hoa thường theo biển số
                query["license_plate"] = {"$regex": search, "$options": "i"}
            if v_type:
                # Vượt đèn đỏ hoặc Không đội mũ
                query["violation_type"] = v_type
            if status:
                query["status"] = status
                
            # Đếm tổng
            total = await collection.count_documents(query)
            
            # Lấy danh sách phân trang, sắp xếp theo thời gian tạo mới nhất
            cursor = collection.find(query).sort("timestamp", -1).skip(skip).limit(limit)
            records = []
            async for doc in cursor:
                # Đổi trường _id thành id dạng string để frontend dễ xử lý
                doc["id"] = doc["_id"]
                records.append(doc)
            return records, total
        except Exception as e:
            logger.error(f"❌ [DATABASE] Lỗi truy vấn MongoDB: {e}. Sử dụng dữ liệu Fallback.")

    # Fallback truy vấn trong RAM
    filtered = _fallback_violations
    if search:
        s_upper = search.upper()
        filtered = [v for v in filtered if s_upper in v["license_plate"].upper()]
    if v_type:
        filtered = [v for v in filtered if v["violation_type"] == v_type]
    if status:
        filtered = [v for v in filtered if v["status"] == status]
        
    total = len(filtered)
    # Cắt mảng để phân trang
    records = filtered[skip : skip + limit]
    for r in records:
        r["id"] = r["_id"]
    return records, total


async def update_violation_status(violation_id: str, new_status: str) -> bool:
    """
    Cập nhật trạng thái đóng phạt nguội của một biên bản.
    """
    if new_status not in ("pending", "processed", "paid"):
        return False
        
    if MONGO_ENABLED and collection is not None:
        try:
            res = await collection.update_one(
                {"_id": violation_id},
                {"$set": {"status": new_status}}
            )
            logger.info(f"💼 [DATABASE] Đã cập nhật trạng thái vi phạm {violation_id} -> [{new_status}].")
            return res.modified_count > 0
        except Exception as e:
            logger.error(f"❌ [DATABASE] Lỗi cập nhật trạng thái MongoDB: {e}.")
            
    # Fallback cập nhật trong RAM
    for v in _fallback_violations:
        if v["_id"] == violation_id:
            v["status"] = new_status
            logger.info(f"💼 [DATABASE] [FALLBACK] Đã cập nhật trạng thái vi phạm {violation_id} -> [{new_status}] trong RAM.")
            return True
    return False


async def get_violation_stats() -> Dict[str, Any]:
    """
    Lấy số liệu thống kê tổng hợp để hiển thị biểu đồ hoặc dashboard.
    """
    if MONGO_ENABLED and collection is not None:
        try:
            total_count = await collection.count_documents({})
            redlight_count = await collection.count_documents({"violation_type": "Vượt đèn đỏ"})
            helmet_count = await collection.count_documents({"violation_type": "Không đội mũ"})
            
            pending_count = await collection.count_documents({"status": "pending"})
            processed_count = await collection.count_documents({"status": "processed"})
            paid_count = await collection.count_documents({"status": "paid"})
            
            return {
                "total": total_count,
                "redlight": redlight_count,
                "helmet": helmet_count,
                "status_stats": {
                    "pending": pending_count,
                    "processed": processed_count,
                    "paid": paid_count
                },
                "db_status": "MongoDB Atlas Cloud Connected"
            }
        except Exception as e:
            logger.error(f"❌ [DATABASE] Lỗi thống kê MongoDB: {e}.")

    # Fallback thống kê trong RAM
    total_count = len(_fallback_violations)
    redlight_count = sum(1 for v in _fallback_violations if v["violation_type"] == "Vượt đèn đỏ")
    helmet_count = sum(1 for v in _fallback_violations if v["violation_type"] == "Không đội mũ")
    
    pending_count = sum(1 for v in _fallback_violations if v["status"] == "pending")
    processed_count = sum(1 for v in _fallback_violations if v["status"] == "processed")
    paid_count = sum(1 for v in _fallback_violations if v["status"] == "paid")
    
    return {
        "total": total_count,
        "redlight": redlight_count,
        "helmet": helmet_count,
        "status_stats": {
            "pending": pending_count,
            "processed": processed_count,
            "paid": paid_count
        },
        "db_status": "RAM Fallback Mode (Demo)"
    }
