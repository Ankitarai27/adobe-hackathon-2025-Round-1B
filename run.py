import os
import json
import fitz  # PyMuPDF
import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

def load_input(input_path):
    with open(input_path, 'r') as f:
        return json.load(f)

def extract_text_by_page(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        text = doc[i].get_text()
        if text.strip():
            pages.append({"page_number": i + 1, "text": text.strip()})
    return pages

def chunk_and_rank(pages, doc_name, query, top_k=5):
    query_vec = model.encode([query])[0]
    scored_chunks = []

    for page in pages:
        for chunk in page['text'].split('\n\n'):
            clean_chunk = chunk.strip()
            if len(clean_chunk) < 50:
                continue
            chunk_vec = model.encode([clean_chunk])[0]
            score = cosine_similarity([query_vec], [chunk_vec])[0][0]
            scored_chunks.append({
                "document": doc_name,
                "page_number": page['page_number'],
                "text": clean_chunk,
                "score": float(score)
            })

    top_chunks = sorted(scored_chunks, key=lambda x: x['score'], reverse=True)[:top_k]
    return top_chunks

def run(input_json_path, pdf_folder, output_json_path):
    data = load_input(input_json_path)

    persona = data['persona']['role']
    job = data['job_to_be_done']['task']
    input_docs = data['documents']
    query = f"{persona} | {job}"

    extracted_sections = []
    subsection_analysis = []
    all_chunks = []

    for doc in input_docs:
        filename = doc['filename']
        pdf_path = os.path.join(pdf_folder, filename)

        if not os.path.exists(pdf_path):
            print(f"File not found: {pdf_path}")
            continue

        pages = extract_text_by_page(pdf_path)
        top_chunks = chunk_and_rank(pages, filename, query, top_k=2)  # 2 per doc to ensure diversity
        all_chunks.extend(top_chunks)

    # Now pick top 5 across all
    all_chunks = sorted(all_chunks, key=lambda x: x['score'], reverse=True)[:5]

    for idx, chunk in enumerate(all_chunks):
        extracted_sections.append({
            "document": chunk['document'],
            "section_title": chunk['text'].split("\n")[0][:100],
            "importance_rank": idx + 1,
            "page_number": chunk['page_number']
        })
        subsection_analysis.append({
            "document": chunk['document'],
            "refined_text": chunk['text'],
            "page_number": chunk['page_number']
        })

    output = {
        "metadata": {
            "input_documents": [doc['filename'] for doc in input_docs],
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": datetime.datetime.now().isoformat()
        },
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }

    with open(output_json_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Output saved to {output_json_path}")

if __name__ == "__main__":
    input_json_path = "input.json"  # path to your input JSON
    pdf_folder = "./input_pdfs"     # folder containing PDFs
    output_json_path = "output.json"
    run(input_json_path, pdf_folder, output_json_path)
