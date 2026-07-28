# To Extract the (critical terms) from text like diseases and chemicals
import spacy
import scispacy 

def extract_medical_entities(text):
    print("Loading Medical NLP Model")
    
    nlp = spacy.load("en_ner_bc5cdr_md")
    
    doc = nlp(text)
    
    entities_list = []
    for ent in doc.ents:
        entities_list.append({
            "term": ent.text,
            "category": ent.label_
        })
        
    unique_entities = {v['term']:v for v in entities_list}.values()
    return list(unique_entities)

if __name__ == "__main__":
    sample_text = "The patient was diagnosed with severe Hyperlipidemia and mild Erythema. Prescribed Atorvastatin 20mg."
    
    print("Extracting entities...")
    results = extract_medical_entities(sample_text)
    
    print("\n--- Extracted Medical Terms ---")
    for r in results:
        print(f"- {r['term']} (Category: {r['category']})")