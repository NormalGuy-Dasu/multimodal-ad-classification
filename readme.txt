=============================================================================
THÔNG TIN DỰ ÁN
=============================================================================
Tiếng Việt: Phân loại quảng cáo trực tuyến dựa trên dữ liệu đa phương thức (hình ảnh, văn bản, và siêu dữ liệu).
Tiếng Anh: Multi-modal classification of online advertisements using image, text, and metadata.
Sinh viên thực hiện:
1. Nguyễn Tấn Phong - 522H0139
2. Trần Bá Đạt - 522H0156

=============================================================================
CẤU TRÚC SOURCE CODE
=============================================================================
1. Folder "Source/Training": Chứa dữ liệu (images, json) và các Notebook (.ipynb) dùng để huấn luyện mô hình.
2. Folder "Source/App_Demo": Chứa mã nguồn ứng dụng Web Demo (đã bao gồm các model weights trong thư mục checkpoints).
3. Các file "requirements_*.txt": Danh sách thư viện cần thiết cho từng môi trường.

=============================================================================
HƯỚNG DẪN CÀI ĐẶT MÔI TRƯỜNG (ANACONDA)
=============================================================================
Dự án yêu cầu chạy trên Anaconda Prompt.
Hệ thống yêu cầu chạy trên 3 môi trường ảo (Virtual Environments) riêng biệt để tránh xung đột thư viện.
Vui lòng tạo và cài đặt như sau:

1. Môi trường 1 (Dùng cho OCR Service):
   - Tạo môi trường: conda create -n env_ocr python=3.9
   - Kích hoạt: conda activate env_ocr
   - Di chuyển vào thư mục "source_code" 
   - Cài thư viện: pip install -r requirements_paddleocr_env.txt

2. Môi trường 2 (Dùng cho Generation Service - BLIP):
   - Tạo môi trường: conda create -n env_blip python=3.10
   - Kích hoạt: conda activate env_blip
   - Di chuyển vào thư mục "source_code" 
   - Cài thư viện: pip install -r requirements_blip_env.txt

3. Môi trường 3 (Dùng cho Main App - Torch/CLIP):
   - Tạo môi trường: conda create -n env_main python=3.10
   - Kích hoạt: conda activate env_main
   - Di chuyển vào thư mục "source_code" 
   - Cài thư viện: pip install -r requirements_torch_env.txt

=============================================================================
HƯỚNG DẪN CHẠY DEMO (WEB APP)
=============================================================================
Vui lòng mở 3 cửa sổ Anaconda Prompt riêng biệt, trỏ tất cả về thư mục "source_code/Source/App_Demo" và thực hiện lần lượt:

*** TAB 1: KHỞI ĐỘNG OCR SERVICE ***
1. Kích hoạt môi trường OCR:
   conda activate env_ocr
2. Di chuyển vào thư mục service:
   cd ocr_service
3. Chạy lệnh:
   python server.py

*** TAB 2: KHỞI ĐỘNG GENERATION SERVICE (BLIP) ***
1. Kích hoạt môi trường BLIP:
   conda activate env_blip
2. Di chuyển vào thư mục service:
   cd gen_service
3. Chạy lệnh:
   python server.py

*** TAB 3: KHỞI ĐỘNG MAIN APP (Giao diện chính) ***
1. Kích hoạt môi trường Main/Torch:
   conda activate env_main
2. Di chuyển vào thư mục app:
   cd main_app
3. Chạy lệnh:
   python app.py
-> Sau khi chạy, truy cập đường dẫn hiển thị trên màn hình (ví dụ: http://127.0.0.1:5000) để sử dụng Demo.

=============================================================================
LƯU Ý
=============================================================================
- Cần đảm bảo chạy đúng thứ tự và đúng môi trường cho từng service.
- Các file model (.pth) đã được huấn luyện và đặt sẵn trong thư mục "Source/App_Demo/checkpoints".
- Nếu muốn huấn luyện lại từ đầu, vui lòng sử dụng các file notebook trong thư mục "Source/Training".
- Thư mục "images" vốn chứa nhiều dữ liệu hình ảnh, nên đã được xóa bớt để không gây nặng máy.