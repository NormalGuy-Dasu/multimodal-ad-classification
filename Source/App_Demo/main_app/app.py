# File: main_app/app.py
import gradio as gr
import torch
import requests
import os
from PIL import Image
from transformers import CLIPProcessor, ViltProcessor
# Import cả CLIP và hàm lấy ViLT từ file model_class
from model_class import MultimodalCLIPClassifier, get_vilt_model

# --- CẤU HÌNH ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_PATH = "../checkpoints/clip_qa_cap_slo_ocr.pth"
VILT_PATH = "../checkpoints/vilt_qa_cap_slo_ocr.pth" # Đường dẫn file ViLT

OCR_API_URL = "http://127.0.0.1:8000/ocr"
GEN_API_URL = "http://127.0.0.1:8001/generate"

# Mapping ID sang Tên (Dựa trên list bạn cung cấp)
TOPIC_NAMES = {
    1: "Restaurants, cafe, fast food",
    2: "Chocolate, cookies, candy, ice cream",
    3: "Chips, snacks, nuts, fruit...",
    4: "Seasoning, condiments, ketchup",
    5: "Alcohol",
    6: "Coffee, tea",
    7: "Soda, juice, milk, energy drinks",
    8: "Cars, automobiles",
    9: "Electronics",
    10: "Phone, TV and internet service",
    11: "Financial services",
    12: "Other services",
    13: "Beauty products and cosmetics",
    14: "Healthcare and medications",
    15: "Clothing and accessories",
    16: "Games and toys",
    17: "Home appliances",
    18: "Vacation and travel",
    19: "Media and arts",
    20: "Sports equipment and activities",
    21: "Shopping and retail products",
    22: "Environment, nature, pollution",
    23: "Animals & Pet Care",
    24: "Safety, security and social awareness",
    25: "Smoking, alcohol abuse",
    26: "Unclear or mixed content"
}

# --- 1. LOAD MODELS ---
print(f"Using Device: {DEVICE}")

# A. Load CLIP
print("⏳ Loading CLIP Model...")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
clip_model = MultimodalCLIPClassifier(num_classes=26).to(DEVICE)
try:
    clip_model.load_state_dict(torch.load(CLIP_PATH, map_location=DEVICE, weights_only=True), strict=False)
    clip_model.eval()
    print("✅ CLIP Ready!")
except Exception as e:
    print(f"❌ CLIP Load Error: {e}")

# B. Load ViLT
print("⏳ Loading ViLT Model...")
vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm", use_fast=True)
vilt_model = get_vilt_model(num_classes=26, device=DEVICE)
try:
    # Load weights
    vilt_model.load_state_dict(torch.load(VILT_PATH, map_location=DEVICE, weights_only=True), strict=False)
    vilt_model.eval()
    print("✅ ViLT Ready!")
except Exception as e:
    print(f"❌ ViLT Load Error: {e}")

# --- 2. HELPER FUNCTIONS (API Calls) ---
# (Giữ nguyên như cũ)
def call_ocr_service(image_path):
    try:
        with open(image_path, 'rb') as f:
            response = requests.post(OCR_API_URL, files={'file': f}, timeout=30)
        if response.status_code == 200 and response.json().get("status") == "success":
            return response.json()["text"]
        return ""
    except: return ""

def call_gen_service(image_path, task):
    try:
        with open(image_path, 'rb') as f:
            data = {"task": task}
            response = requests.post(GEN_API_URL, files={'file': f}, data=data, timeout=60)
        if response.status_code == 200:
            return response.json().get("result", "")
        return ""
    except: return ""

# --- 3. PREDICT LOGIC (Cập nhật xử lý chế độ Image Only) ---
def analyze_and_predict(image, user_ocr, user_caption, user_slogan, selected_model, enable_auto_gen):
    if image is None: return "Upload image first!", "", "", "", {}
    
    # Save Temp Image
    os.makedirs("../temp", exist_ok=True)
    temp_path = "../temp/query.jpg"
    image.save(temp_path)
    
    # Kiểm tra xem các ô có trống không
    all_empty = (not user_ocr.strip()) and (not user_caption.strip()) and (not user_slogan.strip())

    # --- LOGIC QUAN TRỌNG ---
    # Chỉ Auto-fill khi: (Tất cả đều trống) VÀ (Checkbox Auto-gen đang BẬT)
    if all_empty and enable_auto_gen:
        final_ocr = call_ocr_service(temp_path)
        final_caption = call_gen_service(temp_path, "caption")
        final_slogan = call_gen_service(temp_path, "slogan")
    else:
        # Nếu Checkbox tắt (Chế độ Image Only) -> Giữ nguyên text rỗng
        final_ocr = user_ocr
        final_caption = user_caption
        final_slogan = user_slogan

    # Nếu chạy chế độ Image Only, full_text sẽ rỗng -> Gán label mặc định để tránh lỗi model
    full_text = f"{final_caption} {final_slogan} {final_ocr}".strip()
    if not full_text: full_text = "image" 
    
    probs = None
    
    # --- Step 2: Inference ---
    if selected_model == "CLIP (Original)":
        inputs = clip_processor(
            text=[full_text], images=image, return_tensors="pt", 
            padding="max_length", truncation=True, max_length=77
        )
        inputs["input_ids"] = inputs["input_ids"].unsqueeze(1)
        inputs["attention_mask"] = inputs["attention_mask"].unsqueeze(1)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = clip_model(inputs["pixel_values"], inputs["input_ids"], inputs["attention_mask"])
            probs = torch.softmax(logits, dim=1)[0]

    else: # ViLT
        inputs = vilt_processor(
            images=image, text=full_text, return_tensors="pt",
            padding="max_length", truncation=True, max_length=40
        )
        
        # --- FIX LỖI DIMENSION ---
        # Loại bỏ pixel_mask vì nó gây lỗi dimension (3D vs 4D conflict)
        if "pixel_mask" in inputs:
            inputs.pop("pixel_mask")
            
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()} # Đẩy data sang GPU/CPU sau khi pop
        
        with torch.no_grad():
            outputs = vilt_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0]

    # --- Step 3: Result ---
    top5_prob, top5_idx = torch.topk(probs, 5)
    results = {}
    for i in range(5):
        topic_name = TOPIC_NAMES.get(top5_idx[i].item() + 1, "Unknown")
        results[topic_name] = top5_prob[i].item()
        
    return full_text, final_ocr, final_caption, final_slogan, results

# --- 4. INTERFACE ---
with gr.Blocks(title="Ads Classifier Ultimate") as demo:
    gr.Markdown("# 🚀 Ads Classifier: CLIP vs ViLT")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Ảnh Quảng Cáo")
            
            model_selector = gr.Radio(
                ["CLIP (Original)", "ViLT (New)"], 
                label="Chọn Model Dự Đoán", 
                value="CLIP (Original)"
            )

            # --- CHECKBOX CHẾ ĐỘ IMAGE ONLY ---
            # Dùng Checkbox để có trạng thái True/False rõ ràng
            chk_image_only = gr.Checkbox(
                label="👁️ Chế độ chỉ dùng Ảnh (Image Only)", 
                value=False, # Mặc định không tích
                info="Tích vào để ẩn Text và tắt tính năng tự động tạo nội dung."
            )
            
            # --- Accordion Text ---
            # Gán vào biến acc_text_group để code có thể điều khiển đóng/mở
            with gr.Accordion("Chi tiết Text (AI Generated)", open=True) as acc_text_group:
                # Checkbox con (ẩn hoặc đồng bộ logic)
                chk_auto_gen = gr.Checkbox(label="Tự động tạo Text nếu thiếu", value=True, visible=False) 
                
                with gr.Row():
                    txt_slogan = gr.Textbox(label="Slogan", placeholder="...", lines=2, max_lines=5, scale=8)
                    btn_cls_slogan = gr.Button("❌", scale=1, min_width=10)
                with gr.Row():
                    txt_caption = gr.Textbox(label="Caption", placeholder="...", lines=3, max_lines=10, scale=8)
                    btn_cls_caption = gr.Button("❌", scale=1, min_width=10)
                with gr.Row():
                    txt_ocr = gr.Textbox(label="OCR", placeholder="...", lines=4, max_lines=20, scale=8)
                    btn_cls_ocr = gr.Button("❌", scale=1, min_width=10)
                
                btn_clear_all = gr.Button("🗑️ Xóa hết Text (Reset)", variant="secondary")
            
            btn_run = gr.Button("🔥 PHÂN TÍCH", variant="primary")
            
        with gr.Column(scale=1):
            lbl_result = gr.Label(num_top_classes=5, label="Top Prediction")
            debug_text = gr.Textbox(label="Full Input Text", lines=3)

    # --- HÀM XỬ LÝ LOGIC UI ---
    
    def toggle_mode(is_image_only):
        """
        Hàm này chạy khi người dùng tích/bỏ tích Checkbox Image Only.
        Nó trả về trạng thái mới cho Accordion và các ô text.
        """
        if is_image_only:
            # Nếu ĐANG TÍCH (Chế độ ảnh):
            # 1. Đóng Accordion (open=False)
            # 2. Xóa sạch text trong 3 ô
            # 3. Tắt auto-gen (để logic phân tích biết là không được gọi AI)
            return gr.Accordion(open=False), "", "", "", False
        else:
            # Nếu BỎ TÍCH (Chế độ thường):
            # 1. Mở Accordion (open=True)
            # 2. Giữ nguyên text (hoặc trả về placeholder, ở đây ta để nguyên dùng gr.update())
            # 3. Bật lại auto-gen
            # Lưu ý: gr.update() giữ nguyên giá trị cũ nếu không truyền value
            return gr.Accordion(open=True), gr.update(), gr.update(), gr.update(), True

    # --- SỰ KIỆN ---

    # 1. Sự kiện khi bấm Checkbox "Chế độ chỉ dùng Ảnh"
    chk_image_only.change(
        fn=toggle_mode,
        inputs=[chk_image_only],
        outputs=[acc_text_group, txt_slogan, txt_caption, txt_ocr, chk_auto_gen]
    )

    # 2. Các nút xóa lẻ (Giữ nguyên)
    btn_cls_slogan.click(fn=lambda: "", inputs=None, outputs=txt_slogan)
    btn_cls_caption.click(fn=lambda: "", inputs=None, outputs=txt_caption)
    btn_cls_ocr.click(fn=lambda: "", inputs=None, outputs=txt_ocr)

    # 3. Nút Xóa hết (Reset) -> Cũng phải đảm bảo Textbox Image Only bị bỏ tích
    btn_clear_all.click(
        fn=lambda: ("", "", "", True, False, gr.Accordion(open=True)),
        inputs=None, 
        outputs=[txt_slogan, txt_caption, txt_ocr, chk_auto_gen, chk_image_only, acc_text_group]
    )

    # 4. Nút chạy chính
    # Cần truyền chk_image_only vào hàm predict để logic biết
    # (Lưu ý: Logic predict cần sửa nhẹ để dùng chk_image_only thay vì chk_auto_gen nếu muốn code sạch hơn,
    # nhưng ở đây mình dùng chk_auto_gen (được sync ngầm) để tái sử dụng code cũ).
    btn_run.click(
        analyze_and_predict,
        inputs=[input_img, txt_ocr, txt_caption, txt_slogan, model_selector, chk_auto_gen],
        outputs=[debug_text, txt_ocr, txt_caption, txt_slogan, lbl_result]
    )

if __name__ == "__main__":
    demo.launch()