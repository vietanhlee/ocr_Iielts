import streamlit as st
import os
from PIL import Image
from typing import Dict
import ast

def parse_result_string(result_str: str) -> Dict:
    """Chuyển đổi string kết quả thành dictionary."""
    try:
        # Loại bỏ np.str_() wrapper
        cleaned = result_str.replace("np.str_(", "").replace(")", "")
        return ast.literal_eval(cleaned)
    except Exception:
        return {}

# Lazy loading OCR engines
@st.cache_resource
def load_easyocr():
    """Load EasyOCR model khi cần."""
    import easyocr
    return easyocr.Reader(['en'])

@st.cache_resource
def load_paddleocr(
    text_detection_model_name: str = "PP-OCRv5_server_det",
    text_recognition_model_name: str = "PP-OCRv5_mobile_rec",
    text_recognition_batch_size: int = 8,
    use_doc_orientation_classify: bool = False,
    use_doc_unwarping: bool = False,
    use_textline_orientation: bool = False,
    text_det_unclip_ratio: float = 1.2,
    textline_orientation_batch_size: int = 8,
    text_det_box_thresh: float = 0.7,
    text_det_thresh: float = 0.3,
):
    """Load PaddleOCR model khi cần (được cache theo tham số)."""
    from paddleocr import PaddleOCR
    return PaddleOCR(
        text_detection_model_name=text_detection_model_name,
        text_recognition_model_name=text_recognition_model_name,
        text_recognition_batch_size=text_recognition_batch_size,
        use_doc_orientation_classify=use_doc_orientation_classify,
        use_doc_unwarping=use_doc_unwarping,
        use_textline_orientation=use_textline_orientation,
        text_det_unclip_ratio=text_det_unclip_ratio,
        textline_orientation_batch_size=textline_orientation_batch_size,
        text_det_box_thresh=text_det_box_thresh,
        text_det_thresh=text_det_thresh,
    )

def process_with_easyocr(image_paths):
    """Xử lý OCR bằng EasyOCR."""
    from utils import post_process
    reader = load_easyocr()
    ocr_results = {}
    for image_path in image_paths:
        result = reader.readtext(image_path)
        texts = [res[1] for res in result]
        ocr_results[image_path] = str(post_process(texts))
    return ocr_results

def process_with_paddleocr(image_paths, paddle_params: Dict):
    """Xử lý OCR bằng PaddleOCR."""
    import numpy as np
    from utils import post_process
    ocr = load_paddleocr(**paddle_params)
    results = ocr.predict(input=image_paths)
    ocr_results = {}
    for result in results:
        ocr_results[result["input_path"]] = str(post_process(np.array(result['rec_texts'])))
    return ocr_results

# Giao diện Streamlit
st.set_page_config(
    page_title="OCR IELTS Certificate", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: white !important;
        margin: 0;
        font-size: 2.5rem;
    }
    .main-header p {
        color: rgba(255, 255, 255, 0.95) !important;
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    
    /* Metrics */
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }
    .stMetric label {
        color: #495057 !important;
        font-weight: 600;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #0066cc !important;
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    /* Result cards */
    .result-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    .result-card h3 {
        color: #212529 !important;
        margin-bottom: 1rem;
    }
    
    /* Text colors */
    h1, h2, h3, h4, h5, h6 {
        color: #212529 !important;
    }
    p {
        color: #495057 !important;
    }
    span {
        color: #495057 !important;
    }
    div {
        color: #495057 !important;
    }
    label {
        color: #495057 !important;
    }
    
    /* Main content area */
    .main .block-container {
        background-color: #ffffff;
    }
    .main h1, .main h2, .main h3, .main h4 {
        color: #212529 !important;
    }
    .main p, .main span, .main div, .main li {
        color: #495057 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] button {
        color: #495057 !important;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #212529 !important;
    }
    .stTabs [data-baseweb="tab-panel"] * {
        color: #495057 !important;
    }
    .stTabs [data-baseweb="tab-panel"] h3,
    .stTabs [data-baseweb="tab-panel"] h4 {
        color: #212529 !important;
    }
    
    /* File uploader */
    .stFileUploader label {
        color: #212529 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    .stFileUploader small {
        color: #6c757d !important;
    }
    
    /* Welcome section */
    .welcome-text {
        text-align: center;
        padding: 3rem 0;
    }
    .welcome-text h2 {
        color: #212529 !important;
        margin-bottom: 1rem;
    }
    .welcome-text p {
        font-size: 1.2rem;
        color: #6c757d !important;
    }
    
    /* Info boxes */
    .stInfo, .stSuccess, .stWarning, .stError {
        color: #212529 !important;
    }
    
    /* Markdown text */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] div {
        color: #495057 !important;
    }
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4 {
        color: #212529 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        color: #212529 !important;
    }
    .streamlit-expanderContent * {
        color: #495057 !important;
    }
    
    /* Status container */
    [data-testid="stStatusWidget"] * {
        color: #495057 !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: #212529 !important;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: #495057 !important;
    }
    section[data-testid="stSidebar"] label {
        color: #212529 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * {
        color: #495057 !important;
    }
    
    /* Radio buttons in sidebar */
    section[data-testid="stSidebar"] .stRadio label {
        color: #212529 !important;
    }
    section[data-testid="stSidebar"] .stRadio p {
        color: #495057 !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🎓 OCR IELTS Certificate Reader</h1>
    <p>Trích xuất thông tin tự động từ chứng chỉ IELTS</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Cài đặt OCR")
    
    # Chọn OCR engine
    ocr_engine = st.radio(
        "Chọn công cụ OCR:",
        options=["PaddleOCR", "EasyOCR"],
        help="PaddleOCR: Nhanh, tối ưu cho batch\nEasyOCR: Nhẹ, linh hoạt"
    )
    
    # Tuỳ chọn cho PaddleOCR
    paddle_params = None
    if ocr_engine == "PaddleOCR":
        with st.expander("🔧 Tuỳ chọn PaddleOCR", expanded=False):
            det_model = st.selectbox(
                "Model phát hiện (det)",
                options=["PP-OCRv5_server_det", "PP-OCRv4_server_det"],
                index=0
            )
            rec_model = st.selectbox(
                "Model nhận dạng (rec)",
                options=["PP-OCRv5_mobile_rec", "PP-OCRv4_mobile_rec"],
                index=0
            )
            rec_bs = st.number_input(
                "text_recognition_batch_size",
                min_value=1, max_value=64, value=16, step=1
            )
            use_doc_orientation = st.checkbox("use_doc_orientation_classify", value=False)
            use_unwarp = st.checkbox("use_doc_unwarping", value=False)
            use_textline_orient = st.checkbox("use_textline_orientation", value=False)
            det_unclip = st.number_input(
                "text_det_unclip_ratio",
                min_value=0.1, max_value=5.0, value=1.2, step=0.1
            )
            textline_bs = st.number_input(
                "textline_orientation_batch_size",
                min_value=1, max_value=64, value=16, step=1
            )
            det_box_thresh = st.number_input(
                "text_det_box_thresh",
                min_value=0.0, max_value=1.0, value=0.7, step=0.01
            )
            det_thresh = st.number_input(
                "text_det_thresh",
                min_value=0.0, max_value=1.0, value=0.3, step=0.01
            )

            paddle_params = dict(
                text_detection_model_name=det_model,
                text_recognition_model_name=rec_model,
                text_recognition_batch_size=rec_bs,
                use_doc_orientation_classify=use_doc_orientation,
                use_doc_unwarping=use_unwarp,
                use_textline_orientation=use_textline_orient,
                text_det_unclip_ratio=det_unclip,
                textline_orientation_batch_size=textline_bs,
                text_det_box_thresh=det_box_thresh,
                text_det_thresh=det_thresh,
            )
    
    st.markdown("---")
    
    st.markdown("### 📊 Thông tin")
    st.info(f"""
    **Engine:** {ocr_engine}
    
    **Trường trích xuất:**
    - 📅 Ngày thi
    - 👤 Họ và Tên
    - 🆔 Mã thí sinh
    - 🎂 Ngày sinh
    - ⚧ Giới tính
    - 🏆 Band điểm
    - 📅 Ngày cấp
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Lưu ý")
    st.warning("Ảnh nên rõ nét và đầy đủ thông tin để có kết quả tốt nhất")

# Upload section
uploaded_files = st.file_uploader(
    "📁 Chọn ảnh chứng chỉ IELTS",
    type=['png', 'jpg', 'jpeg'],
    accept_multiple_files=True,
    help="Có thể chọn nhiều ảnh cùng lúc"
)

if uploaded_files:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.success(f"✅ Đã tải lên {len(uploaded_files)} ảnh")
    with col2:
        start_button = st.button("🚀 Bắt đầu OCR", type="primary", width='stretch')
    
    if start_button:
        # Tạo thư mục tạm để lưu ảnh
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Progress tracking
        progress_container = st.container()
        
        with progress_container:
            with st.status("🔄 Đang xử lý...", expanded=True) as status:
                st.write("📁 Đang lưu ảnh...")
                
                # Lưu tất cả ảnh vào thư mục tạm
                image_paths = []
                for uploaded_file in uploaded_files:
                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    image_paths.append(temp_path)
                
                # Xử lý OCR theo batch
                st.write(f"🔍 Đang xử lý với {ocr_engine}...")
                
                if ocr_engine == "PaddleOCR":
                    raw_results = process_with_paddleocr(image_paths, paddle_params)
                else:
                    raw_results = process_with_easyocr(image_paths)
                
                # Chuyển đổi kết quả từ string sang dict
                results = {}
                for file_path, result_str in raw_results.items():
                    filename = os.path.basename(file_path)
                    results[filename] = parse_result_string(result_str)
                
                status.update(label="✅ Hoàn thành!", state="complete", expanded=False)
        
        st.markdown("---")
        
        # Hiển thị kết quả
        st.markdown("## 📊 Kết quả OCR")
        
        # Tabs để tổ chức kết quả
        tab1, tab2 = st.tabs(["📋 Xem chi tiết", "📥 Xuất dữ liệu"])
        
        with tab1:
            # Hiển thị từng kết quả
            for idx, (filename, data) in enumerate(results.items(), 1):
                with st.container():
                    st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
                    
                    st.markdown(f"### 📄 Kết quả #{idx}: {filename}")
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        # Hiển thị ảnh
                        temp_path = os.path.join(temp_dir, filename)
                        if os.path.exists(temp_path):
                            image = Image.open(temp_path)
                            st.image(image, width='stretch', caption=filename)
                    
                    with col2:
                        if data:
                            # Chuẩn hóa giá trị và ghép Họ + Tên
                            def clean(v):
                                return str(v).replace("np.str_(", "").replace(")", "").strip("'\"") if v is not None else ""

                            full_name = (clean(data.get('family name')) + ' ' + clean(data.get('first name'))).strip()

                            # Tạo danh sách hiển thị theo thứ tự mong muốn
                            display_items = [
                                ("📅 Ngày thi", clean(data.get('date'))),
                                ("👤 Họ và Tên", full_name),
                                ("🆔 Mã thí sinh", clean(data.get('candidate id'))),
                                ("🎂 Ngày sinh", clean(data.get('date of birth'))),
                                ("⚧ Giới tính", clean(data.get('sex (m/f)'))),
                                ("🏆 Band điểm", clean(data.get('band'))),
                                ("📅 Ngày cấp", clean(data.get('date end')))
                            ]

                            # Lọc bỏ mục trống
                            display_items = [(k, v) for k, v in display_items if v]

                            metric_cols = st.columns(2)
                            for idx_field, (label, value) in enumerate(display_items):
                                with metric_cols[idx_field % 2]:
                                    st.metric(label, value)
                        else:
                            st.error("❌ Không trích xuất được thông tin từ ảnh này")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    if idx < len(results):
                        st.markdown("---")
        
        with tab2:
            st.markdown("### 💾 Xuất dữ liệu")
            
            # Hiển thị JSON
            st.json(results)
            
            # Nút download
            import json
            json_str = json.dumps(results, ensure_ascii=False, indent=2)
            st.download_button(
                label="📥 Tải xuống JSON",
                data=json_str,
                file_name="ocr_results.json",
                mime="application/json",
                width='stretch'
            )
        
        
        # Dọn dẹp thư mục tạm
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass

else:
    # Empty state với hướng dẫn
    st.markdown("""
    <div class="welcome-text">
        <h2>👋 Chào mừng bạn đến với OCR IELTS Certificate Reader</h2>
        <p>Hãy tải lên ảnh chứng chỉ IELTS để bắt đầu</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Hướng dẫn sử dụng với columns
    st.markdown("### 📖 Hướng dẫn sử dụng")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 1️⃣ Chọn công cụ OCR
        Ở thanh bên trái, chọn:
        - **PaddleOCR**: Nhanh, tối ưu
        - **EasyOCR**: Chính xác hơn
        """)
    
    with col2:
        st.markdown("""
        #### 2️⃣ Tải ảnh lên
        - Nhấn vào ô upload
        - Chọn một hoặc nhiều ảnh
        - Định dạng: PNG, JPG, JPEG
        """)
    
    with col3:
        st.markdown("""
        #### 3️⃣ Xem kết quả
        - Nhấn "Bắt đầu OCR"
        - Xem thông tin trích xuất
        - Tải xuống file JSON
        """)
    
    st.markdown("---")
    
    # Thêm tips
    st.markdown("### 💡 Mẹo để có kết quả tốt nhất")
    tips_col1, tips_col2 = st.columns(2)
    
    with tips_col1:
        st.success("✅ **NÊN:**\n- Ảnh rõ nét, đầy đủ ánh sáng\n- Chụp thẳng góc, không bị nghiêng\n- File JPG hoặc PNG chất lượng cao")
    
    with tips_col2:
        st.error("❌ **TRÁNH:**\n- Ảnh mờ, thiếu sáng\n- Bị che khuất hoặc cắt xén\n- Chất lượng ảnh quá thấp")
