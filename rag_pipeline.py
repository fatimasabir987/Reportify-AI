# To Explain the Critical Terms in Easy Language (English + Urdu) 
import streamlit as st
from groq import Groq

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

TEXT_MODEL = "llama-3.3-70b-versatile" 

EXPLAIN_PROMPT = """
Tum ek medical assistant ho jo patients ko unki prescription/report ki
medical terminology asaan zaban mein samjhate ho.

Term: "{term}"
Detected Category: {category}

Agar yeh genuinely ek medical disease, condition, procedure, ya medicine/chemical
ka naam hai, to 2-3 sentences mein iski simple explanation do:
1. Pehle English mein plain-language explanation.
2. Phir ek chhota Urdu tarjuma/summary (Roman Urdu chalega).

Agar yeh term actually ek date, abbreviation, ya non-medical cheez hai
(jaise "May/12" ek date hai, koi disease nahi), to sirf yeh likho:
"Not a recognizable medical term - likely a false detection."

Sirf explanation do, koi extra preamble ya headings nahi.
"""

def setup_medical_knowledge_base():
    return None


def explain_term_with_rag(term, category="", vector_store=None):
    try:
        prompt = EXPLAIN_PROMPT.format(term=term, category=category or "Unknown")
        response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Explanation could not be generated: {e}"


if __name__ == "__main__":
    print(explain_term_with_rag("Hyperlipidemia", "DISEASE"))
