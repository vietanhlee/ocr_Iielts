import streamlit as st
import os
from PIL import Image
from typing import Dict
import ast
from main import process_ocr

def parse_result_string(result_str: str) -> Dict:
    """Chuyển đổi string kết quả thành dictionary."""
    try:
        # Loại bỏ np.str_() wrapper
        cleaned = result_str.replace("np.str_(", "").replace(")", "")
        return ast.literal_eval(cleaned)
    except:
        return {}

# Giao diện Streamlit
st.set_page_config(page_title="OCR IELTS Certificate", layout="wide")
st.title("🎓 OCR IELTS Certificate Reader")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Cài đặt")
    st.info("Tải lên ảnh chứng chỉ IELTS để trích xuất thông tin")

# Upload files
uploaded_files = st.file_uploader(
    "Chọn ảnh chứng chỉ IELTS",
    type=['png', 'jpg', 'jpeg'],
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"Đã tải lên {len(uploaded_files)} ảnh")
    
    if st.button("🚀 Bắt đầu OCR", type="primary"):
        # Tạo thư mục tạm để lưu ảnh
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        
        status_text = st.empty()
        status_text.text("📁 Đang lưu ảnh...")
        
        # Lưu tất cả ảnh vào thư mục tạm
        image_paths = []
        for uploaded_file in uploaded_files:
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            image_paths.append(temp_path)
        
        # Xử lý OCR theo batch (tất cả ảnh cùng lúc)
        status_text.text("🔄 Đang xử lý OCR batch...")
        with st.spinner("Đang xử lý..."):
            raw_results = process_ocr(image_paths)
        
        # Chuyển đổi kết quả từ string sang dict
        results = {}
        for file_path, result_str in raw_results.items():
            filename = os.path.basename(file_path)
            results[filename] = parse_result_string(result_str)
        
        status_text.text("✅ Hoàn thành!")
        st.markdown("---")
        
        # Hiển thị kết quả
        st.header("📊 Kết quả OCR")
        
        # Hiển thị dạng bảng
        for filename, data in results.items():
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader(f"📄 {filename}")
                # Hiển thị ảnh
                temp_path = os.path.join(temp_dir, filename)
                if os.path.exists(temp_path):
                    image = Image.open(temp_path)
                    st.image(image, use_container_width=True)
            
            with col2:
                st.subheader("Thông tin trích xuất")
                if data:
                    # Hiển thị dạng bảng đẹp
                    field_names = {
                        'date': '📅 Ngày',
                        'family name': '👤 Họ',
                        'first name': '👤 Tên',
                        'candidate id': '🆔 Mã thí sinh',
                        'date of birth': '🎂 Ngày sinh',
                        'sex (m/f)': '⚧ Giới tính',
                        'band': '🏆 Band điểm'
                    }
                    
                    for key, value in data.items():
                        # Loại bỏ np.str_() nếu có
                        clean_value = str(value).replace("np.str_(", "").replace(")", "").strip("'\"")
                        display_name = field_names.get(key, key.title())
                        st.metric(display_name, clean_value)
                else:
                    st.warning("Không trích xuất được thông tin")
            
            st.markdown("---")
        
        # Hiển thị JSON raw
        with st.expander("🔍 Xem dữ liệu JSON"):
            st.json(results)
        
        # Dọn dẹp thư mục tạm
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass

else:
    st.info("👆 Hãy tải lên ảnh chứng chỉ IELTS để bắt đầu")
    
    # Hiển thị demo
    st.markdown("### 📝 Hướng dẫn sử dụng")
    st.markdown("""
    1. Nhấn nút **Browse files** để chọn ảnh
    2. Có thể chọn nhiều ảnh cùng lúc
    3. Nhấn **Bắt đầu OCR** để xử lý
    4. Xem kết quả được hiển thị bên dưới
    """)
