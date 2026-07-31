# Reportify AI - Smart Medical Report Explainer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://reportify-ai-fs.streamlit.app/)

Reportify AI is a cloud-native, intelligent web application designed to help patients understand their complex medical reports and handwritten prescriptions. It extracts clinical jargon from uploaded PDFs and provides simplified, plain-language explanations in **English and Roman Urdu**.

### Live Demo
**Try the app here:** [https://reportify-ai-fs.streamlit.app/](https://reportify-ai-fs.streamlit.app/)

---

## Features

* **Intelligent Document Ingestion:** Supports both digital PDFs and scanned/handwritten documents.
* **Advanced OCR for Clinical Shorthand:** Uses Groq's Vision AI (`qwen-vision`) to transcribe difficult doctor handwriting and clinical shorthand (e.g., O/E, Hx, LEEP).
* **LLM-Based Medical NER:** Utilizes `LLaMA-3.3-70B` via Groq to accurately extract specific medical entities, classifying them into Diseases, Chemicals, Procedures, and Clinical Abbreviations.
* **Bilingual Explanations:** Explains complex terms directly using Zero-Shot Prompting, providing a 2-3 line English summary followed by a quick Roman Urdu translation.
* **Ultra-Lightweight & Cloud-Native:** Completely serverless and stateless architecture with zero heavy local dependencies (No local vector databases or heavy NLP models).

## Tech Stack

* **Frontend Hosting:** Streamlit Community Cloud
* **PDF Processing:** PyMuPDF (`fitz`), `pdfplumber`
* **AI Inference (OCR & LLM):** Groq API
* **Models Used:** 
  * `qwen/qwen3.6-27b` (For OCR fallback on scanned images)
  * `llama-3.3-70b-versatile` (For Entity Extraction and Explanation generation)

## Project Structure

* `app.py`: The main Streamlit application and UI frontend.
* `data_ingestion.py`: Handles PDF parsing and Vision AI OCR for scanned/handwritten files.
* `medical_ner.py`: LLM-based Named Entity Recognition to extract relevant clinical terms.
* `llm_pipeline.py`: Direct LLM inference pipeline generating bilingual explanations with built-in medical hallucination guardrails.
* `requirements.txt`: Minimal dependencies required to run the app.
