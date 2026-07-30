# To Extract the text from the image or the pdf

import pdfplumber
import fitz
from PIL import Image
import io
import os
import base64
import re
import streamlit as st
from groq import Groq

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

VISION_MODEL = "qwen/qwen3.6-27b"

PRESCRIPTION_PROMPT = """
Tum ek medical OCR assistant ho jo doctors ke clinical shorthand notes padhne
mein specialize karte ho. Is image mein ek handwritten ya printed
prescription/report hai. Jitna bhi text nazar aaye, accurately transcribe karo.

IMPORTANT - Clinical shorthand mein ye patterns bohot common hain, in par
khaas dhyan do (especially thin slash "/" strokes jo missed ho sakte hain):
- O/E = On Examination
- Hx = History, Dx = Diagnosis, Tx = Treatment, Rx = Prescription
- P/P = Pelvic/Pap, F/U = Follow Up, c̄ (line over c) = "with"
- s̄ (line over s) = "without"

Agar koi 2-3 letter ka short abbreviation dikhe jismein beech mein ek
missing/faint stroke ho sakta hai (jaise "OE" ya "OLE"), socho ke kya ye
ek slash ke sath common shorthand ho sakta hai (jaise "O/E").

Rules:
- Sirf transcription do, koi extra commentary ya explanation nahi.
- Agar koi word clearly samajh na aaye, [unclear] likh do us jagah.
- Original formatting/line breaks ke jitna qareeb ho sako rakho.
- Slashes (/), dashes (-), aur lines-over-letters (macrons) ko khaas ghor
  se dekho - ye clinical notes mein meaning completely badal dete hain.
"""


def extract_text_from_pdf(pdf_file_path):
    """
    Ye function uploaded PDF report se text extract karega.
    Pehle digital text nikalne ki koshish karta hai (pdfplumber).
    Agar text na mile (scanned / handwritten PDF), Groq Vision OCR pe fallback karta hai.
    """
    extracted_text = ""
    try:
        with pdfplumber.open(pdf_file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"

        extracted_text = extracted_text.strip()

        if not extracted_text:
            extracted_text = extract_text_with_groq_vision(pdf_file_path)

        return extracted_text

    except Exception as e:
        return f"Error extracting text: {e}"


def _call_groq_vision(base64_image, use_hidden_reasoning=True):
    """Single Groq vision call. Returns the text content (may be empty)."""
    kwargs = dict(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PRESCRIPTION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            }
        ],
        temperature=0.2,
        reasoning_effort="none",  
    )
    if use_hidden_reasoning:
        kwargs["reasoning_format"] = "hidden"
    else:
        kwargs["reasoning_format"] = "raw"

    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content or ""

    if not use_hidden_reasoning:
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    return content.strip()


def extract_text_with_groq_vision(pdf_file_path):
    """
    Scanned ya handwritten PDF ke liye Groq Vision OCR fallback.
    Agar 'hidden' reasoning format khaali result de (preview model ka
    known flaky behaviour), to 'raw' format ke sath retry karta hai aur
    reasoning ko manually strip karta hai, taake result kabhi silently
    empty na rahe.
    """
    ocr_text = ""
    try:
        pdf_doc = fitz.open(pdf_file_path)

        for page_num, page in enumerate(pdf_doc, start=1):
            pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
            img_bytes = pix.tobytes("png")
            base64_image = base64.b64encode(img_bytes).decode("utf-8")

            page_text = _call_groq_vision(base64_image, use_hidden_reasoning=True)

            if not page_text:
                page_text = _call_groq_vision(base64_image, use_hidden_reasoning=False)

            if not page_text:
                page_text = f"[Page {page_num}: model returned no text after retry]"

            ocr_text += page_text + "\n"

        pdf_doc.close()
        return ocr_text.strip()

    except Exception as e:
        return f"Error during Groq Vision OCR extraction: {e}"


if __name__ == "__main__":
    print("Data Ingestion Module Ready!")
