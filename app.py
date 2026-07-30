import streamlit as st
import tempfile
import os
from PIL import Image

LOGO_PATH = "logo.png"
page_icon = Image.open(LOGO_PATH) if os.path.exists(LOGO_PATH) else "🏥"

from data_ingestion import extract_text_from_pdf
from medical_ner import extract_medical_entities
from rag_pipeline import setup_medical_knowledge_base, explain_term_with_rag

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Reportify AI",
    page_icon=page_icon,
    layout="centered",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Custom CSS 
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* Overall app background */
        .stApp {
            background: linear-gradient(180deg, #f4f9fb 0%, #eef3f8 100%);
        }

        /* Hide default streamlit chrome that looks cluttered */
        #MainMenu, footer {visibility: hidden;}

        /* Hero header */
        .hero {
            background: linear-gradient(135deg, #0f766e 0%, #14b8a6 55%, #5eead4 100%);
            padding: 2.2rem 2rem;
            border-radius: 18px;
            color: white;
            margin-bottom: 1.6rem;
            box-shadow: 0 10px 30px rgba(15, 118, 110, 0.25);
        }
        .hero h1 {
            font-size: 1.9rem;
            margin-bottom: 0.3rem;
            font-weight: 800;
        }
        .hero p {
            font-size: 1rem;
            opacity: 0.95;
            margin: 0;
        }

        /* Step badges */
        .step-badge {
            display: inline-block;
            background: #ffffff;
            color: #0f766e;
            border: 1px solid #99f6e4;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-right: 0.4rem;
            margin-bottom: 0.6rem;
        }

        /* Upload box container */
        .upload-card {
            background: white;
            padding: 1.4rem;
            border-radius: 16px;
            border: 1px dashed #99f6e4;
            box-shadow: 0 4px 14px rgba(0,0,0,0.04);
            margin-bottom: 1.4rem;
        }

        /* Section title */
        .section-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: #0f172a;
            margin: 1.4rem 0 0.6rem 0;
        }

        /* Term card colors by category */
        .term-card {
            border-radius: 14px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.8rem;
            border-left: 6px solid #0f766e;
            background: white;
            box-shadow: 0 3px 10px rgba(0,0,0,0.05);
        }
        .term-card.disease { border-left-color: #dc2626; }
        .term-card.chemical { border-left-color: #2563eb; }

        .term-title {
            font-weight: 700;
            font-size: 1.02rem;
            color: #0f172a;
        }
        .term-badge {
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 700;
            padding: 0.15rem 0.55rem;
            border-radius: 999px;
            margin-left: 0.5rem;
            vertical-align: middle;
        }
        .term-badge.disease { background: #fee2e2; color: #b91c1c; }
        .term-badge.chemical { background: #dbeafe; color: #1d4ed8; }
        .term-badge.other { background: #f1f5f9; color: #475569; }

        .term-explanation {
            margin-top: 0.4rem;
            color: #334155;
            font-size: 0.93rem;
            line-height: 1.5;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: #0f172a;
        }
        section[data-testid="stSidebar"] * {
            color: #e2e8f0 !important;
        }

        /* ---- Global text-color fix ----
            Streamlit's theme sets a default text color that can clash with our
            light background (white-on-white until selected). Force readable
            colors on every native widget, not just our custom HTML blocks. */
        .stApp, .stApp p, .stApp span, .stApp label, .stApp div {
            color: #0f172a;
        }

        /* File uploader dropzone: give it an explicit dark bg + light text
            so it's readable regardless of theme */
        section[data-testid="stFileUploaderDropzone"] {
            background: #1e293b !important;
            border-radius: 12px;
        }
        section[data-testid="stFileUploaderDropzone"] * {
            color: #f1f5f9 !important;
        }

        /* st.info / st.success / st.warning boxes */
        div[data-testid="stAlert"] {
            color: #0f172a !important;
        }
        div[data-testid="stAlert"] * {
            color: #0f172a !important;
        }

        /* Field / widget labels (e.g. "Upload Medical PDF") */
        div[data-testid="stWidgetLabel"] p {
            color: #0f172a !important;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Sidebar 
# ----------------------------------------------------------------------
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=80)
    st.markdown("### About")
    st.write(
        "This tool reads your lab report or prescription and explains "
        "complex medical terms in simple **English** and **Urdu**."
    )
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown(
        "1. Upload a PDF report\n"
        "2. We extract the text\n"
        "3. Medical terms are detected\n"
        "4. Each term is explained simply"
    )
    st.markdown("---")
    st.caption("Not a substitute for professional medical advice.")

# ----------------------------------------------------------------------
# Hero header
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>Smart Medical Report Explainer (Reportify AI)</h1>
        <p>Upload your lab report or prescription and get plain-language explanations —
        in English and Urdu — for every complex medical term.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <span class="step-badge">1️⃣ Upload</span>
    <span class="step-badge">2️⃣ Extract</span>
    <span class="step-badge">3️⃣ Detect Terms</span>
    <span class="step-badge">4️⃣ Explain</span>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_backend():
    return setup_medical_knowledge_base()


# 1. Load the database silently in the background
with st.spinner("Warming up the medical knowledge base..."):
    db = load_backend()

# 2. File Uploader UI
st.markdown('<div class="upload-card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload Medical PDF",
    type=["pdf"],
    help="Lab reports, prescriptions, or discharge summaries in PDF format.",
)
st.markdown("</div>", unsafe_allow_html=True)

# Helper to map NER category labels to a css/badge class + friendly name
CATEGORY_STYLE = {
    "DISEASE": ("disease", "Disease / Condition"),
    "CHEMICAL": ("chemical", "Medicine / Chemical"),
    "PROCEDURE": ("chemical", "Medical Procedure"),
    "ABBREVIATION": ("other", "Clinical Shorthand"),
}

def style_for_category(category: str):
    key = category.upper()
    return CATEGORY_STYLE.get(key, ("other", category))


if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    progress = st.progress(0, text="Starting...")

    with st.spinner("Extracting text from document..."):
        raw_text = extract_text_from_pdf(tmp_file_path)
    progress.progress(40, text="Text extracted")

    with st.expander("Show Extracted Raw Text"):
        st.markdown(
            f'<div style="color:#0f172a; background:#f8fafc; padding:0.8rem; '
            f'border-radius:8px; white-space:pre-wrap;">{raw_text}</div>',
            unsafe_allow_html=True,
        )

    # 3. Process Medical Terms
    with st.spinner("Analyzing medical terms..."):
        entities = extract_medical_entities(raw_text)
    progress.progress(80, text="Medical terms detected")

    # Sirf DISEASE aur CHEMICAL category ke terms rakho
    entities = [e for e in entities if e.get("category", "").upper() in ("DISEASE", "CHEMICAL")]

    # 4. Show Explanations
    st.markdown('<div class="section-title">Simplified Explanations</div>', unsafe_allow_html=True)

    if entities:
        progress.progress(100, text="Done")
        for ent in entities:
            term = ent["term"]
            category = ent["category"]
            css_class, friendly_category = style_for_category(category)

            explanation = explain_term_with_rag(term, category, db)

            st.markdown(
                f"""
                <div class="term-card {css_class}">
                    <span class="term-title">{term}</span>
                    <span class="term-badge {css_class}">{friendly_category}</span>
                    <div class="term-explanation">{explanation}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        progress.progress(100, text="Done")
        st.success("No complex medical terms or diseases found in this document!")

    os.remove(tmp_file_path)
else:
    st.info("Upload a PDF above to get started.")
