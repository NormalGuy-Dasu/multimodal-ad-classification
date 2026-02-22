# File: ocr_service/server.py
from fastapi import FastAPI, File, UploadFile
from paddleocr import PaddleOCR
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image


# ===== GPU BATCH SETTINGS =====
REC_BATCH_NUM = 4      # Batch cho text recognition
CLS_BATCH_NUM = 4      # Batch cho angle classification

app = FastAPI()

print("⏳ Loading PaddleOCR engine...")
# lang='en' hỗ trợ tốt cả tiếng Anh và Việt không dấu. 
# Nếu quảng cáo nhiều tiếng Việt có dấu, hãy đổi lang='vi' (sẽ tải model vi về)
ocr_engine = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    show_log=False,
    rec_batch_num=REC_BATCH_NUM,   # ✅ Batch nội bộ cho recognition
    cls_batch_num=CLS_BATCH_NUM    # ✅ Batch nội bộ cho classifier
)
print("✅ PaddleOCR Loaded!")

@app.post("/ocr")
async def get_ocr(file: UploadFile = File(...)):
    try:
        # Đọc ảnh từ request
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert('RGB')
        img_np = np.array(image)

        # Chạy OCR
        result = ocr_engine.ocr(img_np, cls=True)
        
        text_content = ""
        if result and result[0]:
            # Ghép các đoạn text lại với nhau
            text_lines = [line[1][0] for line in result[0]]
            text_content = " ".join(text_lines)
            
        return {"status": "success", "text": text_content}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # Chạy server tại localhost, port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)