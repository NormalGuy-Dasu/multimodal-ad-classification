# File: gen_service/server.py
from fastapi import FastAPI, UploadFile, File, Form
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from peft import PeftModel
import torch
from PIL import Image
import uvicorn
import io

app = FastAPI()

# --- CẤU HÌNH ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL_ID = "Salesforce/blip2-opt-2.7b"
# Đường dẫn tới folder LoRA slogan bạn đã train (đã copy vào folder checkpoints)
LORA_CHECKPOINT = "../checkpoints/blip2-slogan-best-model" 

print(f"⏳ Đang load BLIP-2 Base Model trên {DEVICE}...")
try:
    processor = Blip2Processor.from_pretrained(BASE_MODEL_ID, use_fast=True)
    # Load 8-bit để tiết kiệm VRAM như trong notebook của bạn
    base_model = Blip2ForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID, 
        load_in_8bit=True, 
        device_map="auto"
    )
    print("✅ Base Model Loaded.")
    
    print(f"⏳ Đang nạp LoRA Adapter từ {LORA_CHECKPOINT}...")
    # Wrap model bằng Peft để có tính năng LoRA
    model = PeftModel.from_pretrained(base_model, LORA_CHECKPOINT)
    model.eval()
    print("✅ LoRA Adapter Loaded! Service Ready.")
except Exception as e:
    print(f"❌ Error loading models: {e}")

@app.post("/generate")
async def generate_text(file: UploadFile = File(...), task: str = Form(...)):
    """
    Task: 'caption' (dùng Base Model) hoặc 'slogan' (dùng LoRA)
    """
    try:
        # Đọc ảnh
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        
        # Tiền xử lý
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        
        generated_text = ""
        
        # --- LOGIC THÔNG MINH: CHUYỂN ĐỔI ADAPTER ---
        if task == "caption":
            # Tắt Adapter để dùng kiến thức gốc của BLIP tạo Caption chuẩn
            # Context manager disable_adapter() giúp tạm thời bỏ qua LoRA
            with model.disable_adapter():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    min_length=5,
                    num_beams=3,
                    repetition_penalty=1.2,
                    early_stopping=True
                )
        elif task == "slogan":
            # Mặc định PeftModel sẽ dùng Adapter (LoRA) đang active
            # Config giống file gen_slogan_json.ipynb
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.8,
                top_p=0.9
            )
        else:
            return {"error": "Unknown task. Use 'caption' or 'slogan'."}
            
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        return {"task": task, "result": generated_text}
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Chạy ở port 8001 để không đụng port 8000 của OCR
    uvicorn.run(app, host="127.0.0.1", port=8001)