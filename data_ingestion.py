# To Extract the text from the image or the pdf
import pdfplumber
import fitz
from PIL import Image
import io
import os
import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

gemini_model = genai.GenerativeModel("gemini-2.0-flash")

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
    Agar text na mile (scanned / handwritten PDF), Gemini Vision OCR pe fallback karta hai.
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
            extracted_text = extract_text_with_gemini_vision(pdf_file_path)

        return extracted_text

    except Exception as e:
        return f"Error extracting text: {e}"


def extract_text_with_gemini_vision(pdf_file_path):
    """
    Scanned ya handwritten PDF ke liye Gemini Vision OCR fallback.
    PyMuPDF (fitz) se PDF pages ko images mein render karta hai
    (poppler ki zaroorat nahi), phir Gemini 2.0 Flash (vision) se text nikalta hai.
    Tesseract se bohot behtar hai handwriting ke liye.
    """
    ocr_text = ""
    try:
        pdf_doc = fitz.open(pdf_file_path)

        for page in pdf_doc:
            # zoom = 300/72 for roughly 300 DPI render (better quality for Gemini)
            pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
            page_image = Image.open(io.BytesIO(pix.tobytes("png")))

            response = gemini_model.generate_content([PRESCRIPTION_PROMPT, page_image])
            page_text = response.text if response and response.text else ""

            if page_text:
                ocr_text += page_text + "\n"

        pdf_doc.close()
        return ocr_text.strip()

    except Exception as e:
        return f"Error during Gemini Vision OCR extraction: {e}"


if __name__ == "__main__":
    # sample_text = extract_text_from_pdf("sample_report.pdf")
    # print(sample_text)
    print("Data Ingestion Module Ready!")
