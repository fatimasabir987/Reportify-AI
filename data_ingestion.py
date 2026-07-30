# To Extract the text from the image or the pdf

import pdfplumber
import fitz
from PIL import Image
import io
import os
import base64
import streamlit as st
from groq import Groq

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

VISION_MODEL = "qwen/qwen3.6-27b"  
PRESCRIPTION_PROMPT = """
Tum ek medical OCR assistant ho. Is image mein ek doctor ka handwritten ya printed
prescription/report hai. Jitna bhi text (medicine names, dosage, instructions,
patient details, diagnosis) tumhe nazar aaye, wo accurately transcribe karo.

Rules:
- Sirf transcription do, koi extra commentary ya explanation nahi.
- Agar koi word clearly samajh na aaye, [unclear] likh do us jagah.
- Original formatting/line breaks ke jitna qareeb ho sako rakho.
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


def extract_text_with_groq_vision(pdf_file_path):
    """
    Scanned ya handwritten PDF ke liye Groq Vision OCR fallback.
    PyMuPDF (fitz) se PDF pages ko images mein render karta hai,
    phir Groq ke vision model se text nikalta hai.
    """
    ocr_text = ""
    try:
        pdf_doc = fitz.open(pdf_file_path)

        for page in pdf_doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
            img_bytes = pix.tobytes("png")
            base64_image = base64.b64encode(img_bytes).decode("utf-8")

            response = client.chat.completions.create(
                model=VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PRESCRIPTION_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                temperature=0.2,
            )

            page_text = response.choices[0].message.content
            if page_text:
                ocr_text += page_text + "\n"

        pdf_doc.close()
        return ocr_text.strip()

    except Exception as e:
        return f"Error during Groq Vision OCR extraction: {e}"


if __name__ == "__main__":
    print("Data Ingestion Module Ready!")
