# To Extract the (critical terms) from text like diseases, chemicals, procedures & abbreviations
import json
import re
import streamlit as st
from groq import Groq

client = Groq(api_key=st.secrets["GROQ_API_KEY"])
TEXT_MODEL = "llama-3.3-70b-versatile"

NER_PROMPT = """
Neeche ek doctor ke prescription/clinical note ka text hai. Isme se saare
medically-relevant terms nikaalo - diseases/conditions, medicines/chemicals,
medical procedures, aur clinical abbreviations/shorthand jo patient ke liye
samajhna mushkil ho (jaise LSIL, LEEP, RTC, Pap smear, anovulatory cycles,
endocrine screen, O/E, Hx, waghera).

IMPORTANT: Ye text OCR/vision model se extract hua hai, isliye kabhi kabhi
punctuation missing ho sakta hai (jaise slash "/"). Agar koi term clinical
shorthand lagta ho lekin usmein missing punctuation ho sakta hai, to uska
sabse common clinical meaning socho, raw OCR text ko literally mat lo.
Misaal: agar "OE" ya "OLE" jaisa kuch dikhe standalone examination-related
context mein, iska matlab likely "O/E" (On Examination) hai - ye ek
disease/condition ka naam NAHI hai.

Sirf genuinely medical/clinical terms do - dates, generic words (normal,
will, need), patient names shamil mat karo.

Text:
\"\"\"
{text}
\"\"\"

Sirf JSON array return karo, koi aur text/explanation nahi, exactly is format mein:
[{{"term": "...", "category": "DISEASE"}}, {{"term": "...", "category": "PROCEDURE"}}]
category sirf in mein se ek ho: DISEASE, CHEMICAL, PROCEDURE, ABBREVIATION.
"""


def extract_medical_entities(text):
    """
    Groq LLM se medical terms/procedures/abbreviations extract karta hai.
    (Pehle wala scispacy model sirf formal DISEASE/CHEMICAL PubMed-style
    terms pehchanta tha - clinical shorthand jaisa LSIL, LEEP, RTC use
    nahi kar pata tha.)
    """
    if not text or not text.strip():
        return []
    try:
        response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": NER_PROMPT.format(text=text)}],
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
        json_str = match.group(0) if match else raw
        entities_list = json.loads(json_str)
        unique_entities = {v["term"]: v for v in entities_list if "term" in v}.values()
        return list(unique_entities)
    except Exception as e:
        print(f"Error extracting medical entities: {e}")
        return []


if __name__ == "__main__":
    sample_text = "Had: LEEP May/12, for LSIL. Pap q 1yr. O/E slim. A anovulatory cycles. P Endocrine screen."
    print(extract_medical_entities(sample_text))
