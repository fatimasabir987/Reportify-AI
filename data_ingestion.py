# To Extract the text from the image or the pdf
import pdfplumber
import fitz  
from PIL import Image
import pytesseract
import io


def extract_text_from_pdf(pdf_file_path):
    """
    Ye function uploaded PDF report se text extract karega.
    Pehle digital text nikalne ki koshish karta hai (pdfplumber).
    Agar text na mile (scanned / handwritten PDF), OCR (tesseract) pe fallback karta hai.
    """
    extracted_text = ""
    try:
        with pdfplumber.open(pdf_file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
        extracted_text = extracted_text.strip()

        # Agar digital text nahi mila -> scanned/handwritten PDF, OCR try karo
        if not extracted_text:
            extracted_text = extract_text_with_ocr(pdf_file_path)

        return extracted_text
    except Exception as e:
        return f"Error extracting text: {e}"


def extract_text_with_ocr(pdf_file_path):
    """
    Scanned ya handwritten PDF ke liye OCR fallback.
    PyMuPDF (fitz) se PDF pages ko images mein render karta hai
    (poppler ki zaroorat nahi), phir Tesseract OCR se text nikalta hai.
    """
    ocr_text = ""
    try:
        pdf_doc = fitz.open(pdf_file_path)
        for page in pdf_doc:
            # zoom = 300/72 for roughly 300 DPI render
            pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
            page_image = Image.open(io.BytesIO(pix.tobytes("png")))
            page_text = pytesseract.image_to_string(page_image)
            if page_text:
                ocr_text += page_text + "\n"
        pdf_doc.close()
        return ocr_text.strip()
    except Exception as e:
        return f"Error during OCR extraction: {e}"


if __name__ == "__main__":
    # sample_text = extract_text_from_pdf("sample_report.pdf")
    # print(sample_text)
    print("Data Ingestion Module Ready!")