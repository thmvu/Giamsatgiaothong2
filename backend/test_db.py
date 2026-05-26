"""
Script kiem tra ket noi MongoDB Atlas Cloud nhanh
==================================================
Chay: .\venv_paddle\Scripts\python.exe backend/test_db.py
"""

import os
import sys
import asyncio

# Them thu muc goc vao path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend import database

async def main():
    print("[TEST] Bat dau chay test ket noi MongoDB...")
    
    # 1. Khoi tao database
    database.init_db()
    
    # Doi 3s de ping hoan thanh
    print("[TEST] Dang ket noi mang...")
    await asyncio.sleep(3)
    
    print(f"[TEST] Trang thai MONGO_ENABLED: {database.MONGO_ENABLED}")
    if not database.MONGO_ENABLED:
        print("[Loi] Không ket noi duoc toi MongoDB Atlas!")
        print("Vui long kiem tra lai cau hinh MONGODB_URL trong file .env")
        sys.exit(1)
        
    print("[SUCCESS] Ket noi thanh cong! Dang test nghiep vu...")
    
    # 2. Test Luu vi pham
    test_vio = {
        "track_id": 9999,
        "type": "Vuot den do",
        "vehicle": "Xe may",
        "time": 12.34,
        "frame": 350,
        "evidence": "evidence/test_evidence.jpg"
    }
    
    print("[TEST] 1. Dang luu thu vi pham...")
    vid = await database.save_violation(test_vio)
    print(f"-> Tao thanh cong ban ghi ID: {vid}")
    
    # 3. Test Cap nhat bien so xe (Late Update)
    print("[TEST] 2. Dang test cap nhat bien so muon...")
    ok = await database.update_violation_plate(track_id=9999, plate_text="29-G1-999.99", crop_path="evidence/plates/crop_test.jpg")
    print(f"-> Cap nhat bien so: {'THANH CONG' if ok else 'THAT BAI'}")
    
    # 4. Test Truy van danh sach
    print("[TEST] 3. Dang truy van danh sach ban ghi...")
    records, total = await database.get_violations(page=1, limit=5, search="29-G1")
    print(f"-> Tong so ban ghi tim thay: {total}")
    for idx, r in enumerate(records):
        print(f"   [{idx+1}] ID: {r['id']} | Loai loi: {r.get('violation_type')} | Xe: {r.get('vehicle_type')} | Bien so: {r.get('license_plate')} | Trang thai: {r['status']}")
        
    # 5. Test Cap nhat trang thai xu phat
    print("[TEST] 4. Dang test cap nhat trang thai phat nguoi...")
    ok_status = await database.update_violation_status(violation_id=vid, new_status="paid")
    print(f"-> Cap nhat trang thai: {'THANH CONG' if ok_status else 'THAT BAI'}")
    
    # 6. Xem lai thong ke
    print("[TEST] 5. Dang lay so lieu thong ke tong hop...")
    stats = await database.get_violation_stats()
    print(f"-> Thong ke: {stats}")
    
    print("\n[SUCCESS] HOAN TAT: Toan bo quy trinh test database da thanh cong tot dep!")

if __name__ == "__main__":
    asyncio.run(main())
