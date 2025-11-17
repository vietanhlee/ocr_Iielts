import streamlit as st
import os
import json
import tempfile
from typing import List
import numpy as np
import cv2
import streamlit.components.v1 as components
import base64

try:
    from Paddle import Paddle
except Exception:
    Paddle = None

try:
    # import the local module (file is `Easy.py`)
    from Easy import Easy
except Exception:
    Easy = None


def inject_css():
    css = """
    <style>
    /* Page */
    .app-title {font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; display:flex; align-items:center; gap:12px;}
    .app-banner {background:linear-gradient(90deg,#0f172a,#0b1220); color:white; padding:18px;border-radius:10px;margin-bottom:14px}
    .app-sub {color:#cbd5e1;margin-top:6px}

    /* Thumbnails */
    .thumb-img {border-radius:10px;border:1px solid rgba(0,0,0,0.06);box-shadow:0 6px 18px rgba(2,6,23,0.08);}

    /* Cards grid */
    .cards-row {display:flex;flex-wrap:wrap;gap:12px;align-items:flex-start}
    .card {box-sizing:border-box;flex:1 1 calc(33% - 12px);min-width:200px;max-width:340px;padding:14px;border-radius:10px;border:1px solid #e8eef8;background:linear-gradient(180deg,#ffffff,#fbfdff);box-shadow:0 6px 20px rgba(12,40,80,0.04);}
    .card-title {font-size:16px;margin-bottom:6px;color:#0f172a}
    .card-value {font-size:15px;color:#0b1220}

    /* Buttons */
    .btn {padding:8px 12px;border-radius:8px;border:1px solid #a3b4d5;background:#f1f5fb;cursor:pointer}
    .btn:active {transform:translateY(1px)}
    .btn-success {background:#16a34a;color:white;border-color:transparent}

    /* Small helper */
    .meta {color:#475569;font-size:13px}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def save_uploaded_files(uploaded_files) -> List[str]:
    tmpdir = tempfile.mkdtemp(prefix="st_uploads_")
    paths = []
    for up in uploaded_files:
        ext = os.path.splitext(up.name)[1]
        dest = os.path.join(tmpdir, up.name)
        with open(dest, "wb") as f:
            f.write(up.getbuffer())
        paths.append(dest)
    return paths


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def annotate_and_show(images_annotated: List[np.ndarray], image_paths: List[str], dict_extracted: dict):
    """Display each annotated image larger on the left and compact info cards on the right."""
    for i, img in enumerate(images_annotated):
        if img is None:
            continue
        # make image slightly larger area than info (ratio tuned)
        col_img, col_info = st.columns([1.6, 1])

        # prepare image (BGR->RGB)
        img_rgb = bgr_to_rgb(img)
        h, w = img_rgb.shape[:2]

        # gentle increase: make thumbnail a bit larger but keep quality
        max_thumb_w = 560
        if w > max_thumb_w:
            scale = max_thumb_w / w
            new_w, new_h = int(w * scale), int(h * scale)
            thumb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            # if image is smaller, keep original to avoid upscaling blur
            thumb = img_rgb

        with col_img:
            # render thumbnail as embedded image to apply CSS class
            _, buf = cv2.imencode('.png', cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR))
            b64 = base64.b64encode(buf).decode('utf-8')
            display_w = int(min(thumb.shape[1], max_thumb_w))
            img_html = f"""
<div style='display:flex;flex-direction:column;align-items:flex-start;'>
  <img src='data:image/png;base64,{b64}' class='thumb-img' style='width:{display_w}px; height:auto;' />
  <div class='meta' style='margin-top:8px'>{os.path.basename(image_paths[i])}</div>
</div>
"""
            components.html(img_html, height=min(thumb.shape[0] + 40, 900))

        # lookup extracted data
        key = image_paths[i]
        data = dict_extracted.get(key) or dict_extracted.get(os.path.basename(key))

        with col_info:
            st.markdown("**Extracted Data**")
            parsed = None
            if data is None or data == "":
                parsed = None
            elif isinstance(data, dict):
                parsed = data
            elif isinstance(data, str):
                try:
                    import ast

                    parsed = ast.literal_eval(data)
                    if not isinstance(parsed, dict):
                        parsed = None
                except Exception:
                    parsed = None

            if parsed is None:
                st.write("(no extracted data)")
                continue

            # normalize and prepare items
            norm = {str(k).strip().lower(): v for k, v in parsed.items()}
            icons = {
                "date": "📅",
                "family name": "👤",
                "first name": "🧾",
                "candidate id": "🆔",
                "date of birth": "🎂",
                "sex (m/f)": "⚧",
                "sex": "⚧",
                "band": "⭐",
                "date end": "📌",
            }

            desired = [
                "date",
                "family name",
                "first name",
                "candidate id",
                "date of birth",
                "sex (m/f)",
                "band",
                "date end",
            ]

            items = []
            for k in desired:
                v = None
                if k in norm:
                    v = norm[k]
                elif k == "sex (m/f)":
                    v = norm.get("sex")
                if v is not None:
                    items.append((k, v, icons.get(k, "📌")))

            if not items:
                for k, v in parsed.items():
                    items.append((k, v, "📌"))

            # build cards HTML (inline-block) and center vertically relative to thumbnail height
            # build responsive 3-column cards using flexbox
            card_pieces = []
            for k, v, ico in items:
                piece = (
                    f"<div class='card'>"
                    f"<div class='card-title'>{ico} <strong style='margin-left:8px'>{k}</strong></div>"
                    f"<div class='card-value'>{v}</div>"
                    f"</div>"
                )
                card_pieces.append(piece)

            cards_html = "".join(card_pieces)

            # add a copy button for this image's parsed dict (placed above the cards)
            try:
                json_str = json.dumps(parsed, ensure_ascii=False)
            except Exception:
                json_str = str(parsed)

            copy_html = f"""
<div style='margin-bottom:6px;'>
  <button id='copy_btn_{i}' class='btn' >Copy dict</button>
  <textarea id='copy_txt_{i}' style='display:none;'>{json_str}</textarea>
  <script>
    const btn = document.getElementById('copy_btn_{i}');
    btn.addEventListener('click', async () => {{
      try {{ await navigator.clipboard.writeText(document.getElementById('copy_txt_{i}').value); btn.classList.add('btn-success'); btn.innerText = 'Copied'; setTimeout(()=>{{ btn.classList.remove('btn-success'); btn.innerText='Copy dict'; }},900); }} catch(e) {{ alert('Copy failed') }}
    }});
  </script>
</div>
"""
            components.html(copy_html, height=60)

            # render cards below the copy button; align to top so header "Extracted Data" is close to cards
            outer_html = (
                f"<div style='display:flex;align-items:flex-start;'>"
                f"<div style='width:100%'><div class='cards-row'>{cards_html}</div></div>"
                f"</div>"
            )
            st.markdown(outer_html, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="IELTS OCR Extractor", layout="wide")
    inject_css()
    st.markdown("""
    <div class='app-banner'>
      <div class='app-title'><h2 style='margin:0'>IELTS Score Extraction</h2></div>
      <div class='app-sub'>Upload scanned score reports, run OCR and review extracted fields.</div>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("Settings")
        engine = st.selectbox("OCR Engine", options=["PaddleOCR", "EasyOCR"])
        uploaded_files = st.file_uploader("Upload images (png/jpg/jpeg)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    # prepare image paths from uploaded files (no input/ or output filesystem interactions)
    image_paths = []
    if uploaded_files:
        image_paths = save_uploaded_files(uploaded_files)

    if not image_paths:
        st.info("No images uploaded. Please upload images to run OCR.")

    run = st.button("Run OCR")

    if run and image_paths:
        # instantiate engine
        if engine == "PaddleOCR":
            if Paddle is None:
                st.error("PaddleOCR integration not available. Ensure `Paddle.py` and dependencies are present.")
                return
            ocr = Paddle()
        else:
            if Easy is None:
                st.error("EasyOCR integration not available. Ensure `easy.py` and dependencies are present.")
                return
            ocr = Easy()

        with st.spinner("Running OCR — this may take a while for many images..."):
            try:
                images_annotated, dict_extracted = ocr.predict_multi_and_extract(image_paths)
            except Exception as e:
                st.exception(e)
                return

        st.success("OCR finished")

        # show annotated images and per-image extracted dict
        st.subheader("Annotated Images & Extracted Data")

        # add "Copy All" button to copy aggregated JSON of all extracted dicts
        try:
            aggregated_json = json.dumps(dict_extracted, ensure_ascii=False, indent=2)
        except Exception:
            aggregated_json = str(dict_extracted)

        copy_all_html = f"""
<div style='margin-bottom:8px;'>
  <button id='copy_all_btn' style='padding:8px 12px;border-radius:6px;border:1px solid #666;background:#f0f0f0;cursor:pointer;'>Copy All Dicts</button>
  <textarea id='copy_all_txt' style='display:none;'>{aggregated_json}</textarea>
  <script>
    const btnAll = document.getElementById('copy_all_btn');
    btnAll.addEventListener('click', async () => {{
      try {{ await navigator.clipboard.writeText(document.getElementById('copy_all_txt').value); btnAll.innerText = 'Copied'; setTimeout(()=>btnAll.innerText='Copy All Dicts',900); }} catch(e) {{ alert('Copy failed') }}
    }});
  </script>
</div>
"""
        components.html(copy_all_html, height=60)

        annotate_and_show(images_annotated, image_paths, dict_extracted)

        # (no saving or downloading in this UI version)


if __name__ == "__main__":
    main()
