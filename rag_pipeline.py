# To Explain the Critical Terms in Easy Language
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

def setup_medical_knowledge_base():
    print("Setting up Medical Knowledge Base...")
    
    dictionary_data = [
        "Hyperlipidemia means your blood has too many lipids (fats), such as cholesterol. It increases the risk of heart disease. Urdu: Khoon mein cholesterol ki ziyadti.",
        "Atorvastatin is a medication used to lower bad cholesterol in the blood and prevent strokes. Urdu: Cholesterol kam karne ki dawai.",
        "Erythema is a type of skin rash or redness caused by injured or inflamed blood capillaries. Urdu: Jild par surkh dhabbe."
    ]
    
    documents = [Document(page_content=text) for text in dictionary_data]
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_store = Chroma.from_documents(documents, embeddings)
    
    return vector_store

def explain_term_with_rag(term, vector_store):
    results = vector_store.similarity_search(term, k=1)
    
    if results:
        return results[0].page_content
    else:
        return "Explanation not found in database."

if __name__ == "__main__":
    db = setup_medical_knowledge_base()
    
    test_terms = ["Hyperlipidemia", "Atorvastatin"]
    
    print("\n--- Medical Explanations ---")
    for term in test_terms:
        explanation = explain_term_with_rag(term, db)
        print(f"\nTerm: {term}")
        print(f"Simple Explanation: {explanation}")