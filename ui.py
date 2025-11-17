import streamlit as st


def inject_css(path: str = "assets/styles.css") -> None:
    """Load CSS from a file and inject into the page."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Không tìm thấy file CSS: {path}")


def render_header() -> None:
    """Render the app main header HTML block."""
    st.markdown(
        """
        <div class="main-header">
            <h1>🎓 OCR IELTS Certificate Reader</h1>
            <p>Trích xuất thông tin tự động từ chứng chỉ IELTS</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
