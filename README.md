# 🎓 OCR IELTS Certificate Reader

Công cụ OCR tự động trích xuất thông tin từ chứng chỉ IELTS sử dụng PaddleOCR.

## ✨ Tính năng

- 🔍 Trích xuất tự động các thông tin từ chứng chỉ IELTS:
  - Ngày cấp
  - Họ và tên
  - Mã thí sinh
  - Ngày sinh
  - Giới tính
  - Band điểm
- 📦 Xử lý batch nhiều ảnh cùng lúc
- 🖥️ Giao diện web đơn giản với Streamlit
- 💾 Xuất kết quả ra file JSON

## 📋 Yêu cầu

- Python 3.8+
- Windows/Linux/MacOS

## 🚀 Cài đặt

1. Clone repository hoặc tải về máy

2. Cài đặt dependencies:

```bash
pip install -r requirements.txt
```

## 💻 Sử dụng

### Giao diện Web (Streamlit)

```bash
streamlit run streamlit_app.py
```

- Mở trình duyệt tại `http://localhost:8501`
- Upload ảnh chứng chỉ IELTS
- Nhấn "Bắt đầu OCR"
- Xem kết quả hiển thị ngay trên web

### Command Line

Xử lý tất cả ảnh trong một thư mục:

```bash
python main.py <input_folder> [output_file]
```

**Ví dụ:**

```bash
# Lưu kết quả vào output.json (mặc định)
python main.py input

# Chỉ định file output khác
python main.py input results.json
```

## 📁 Cấu trúc thư mục

```
OCR_paddle/
├── main.py              # Script chính xử lý OCR
├── streamlit_app.py     # Giao diện web Streamlit
├── utils.py             # Hàm tiện ích
├── requirements.txt     # Dependencies
├── README.md
├── input/              # Thư mục chứa ảnh đầu vào
└── output/             # Thư mục lưu kết quả
```

## 📊 Định dạng output

Kết quả được lưu dưới dạng JSON:

```json
{
  "input/1.jpg": "{'date': '26/12/2024', 'family name': 'NGUYEN', 'first name': 'VAN A', ...}",
  "input/2.jpg": "{'date': '26/09/2024', 'family name': 'TRAN', 'first name': 'THI B', ...}"
}
```

## 🔧 Tùy chỉnh

Chỉnh sửa các trường cần trích xuất trong `utils.py`:

```python
key = ['date', 'family name', 'first name', 'candidate id', 'date of birth', 'sex (m/f)', 'band']
```

## 📝 Lưu ý

- Ảnh đầu vào nên rõ nét, không bị mờ
- Hỗ trợ định dạng: PNG, JPG, JPEG
- Model OCR sẽ tự động tải về khi chạy lần đầu

## 🐛 Báo lỗi

Nếu gặp vấn đề, vui lòng kiểm tra:

- Đã cài đặt đúng dependencies
- Ảnh đầu vào có đúng định dạng
- Đường dẫn thư mục chính xác
