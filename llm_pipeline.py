# To Explain the Critical Terms (Disease / Chemical) in Easy Language (English + Urdu) 
import streamlit as st
from groq import Groq

client = Groq(api_key=st.secrets["GROQ_API_KEY"])
TEXT_MODEL = "llama-3.3-70b-versatile"

EXPLAIN_PROMPT = """
Tum ek medical assistant ho jo patients ko unki prescription/report ki
medical terminology asaan zaban mein samjhate ho.

Term: "{term}"
Detected Category: {category}

Agar yeh genuinely ek disease/condition ya medicine/chemical ka naam hai,
to is format mein explanation do:

1. Pehle 2-3 sentences mein PURE ENGLISH mein plain-language explanation.
   Explanation hamesha term ke pure English naam/meaning se start honi
   chahiye (jaise: "LSIL means Low-grade Squamous Intraepithelial Lesion,
   a condition where..."). Koi Urdu/Roman Urdu is English part mein
   bilkul nahi honi chahiye.
2. Phir ek nayi line pe sirf EK chhoti line Roman Urdu mein - sirf mool
   baat ka khulasa, English wali baat ka poora tarjuma nahi (bas 1 line,
   extra detail nahi). Is line se pehle "Urdu:" likh dena.

Agar yeh term actually ek date, abbreviation, procedure, ya non-medical
cheez hai, to sirf yeh likho:
"Not a recognizable medical term - likely a false detection."

IMPORTANT SAFETY RULE: If you are not 100% sure about the medical term, do not guess. Simply state: "I am not entirely sure about this term, please consult your doctor."

Sirf explanation do, koi extra preamble ya headings nahi.
"""

def explain_term_with_llm(term, category=""):
    """
    Direct LLM Inference for Explanation.
    (Vector DB/RAG is completely removed for cloud optimization).
    """
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
    print(explain_term_with_llm("LSIL", "DISEASE"))
