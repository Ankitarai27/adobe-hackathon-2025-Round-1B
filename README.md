## 🚀 How It Works

- Extracts structured data from PDFs using semantic similarity (with sentence-transformers).
- Prioritizes sections relevant to the persona (e.g., Travel Planner) and their task (e.g., “Plan a trip for 10 college friends”).
- Outputs a JSON file highlighting the most useful content for itinerary planning.

## 📥 Input
- input.json includes:
  - Persona
  - Task to be completed
  - List of PDF titles

## 📤 Output
- Metadata
- List of selected sections with:
  - PDF name
  - Section title
  - Importance rank
  - Page number
- Detailed refined text from those sections

## 🧠 Libraries Used
- PyMuPDF for PDF parsing
- sentence-transformers for embeddings
- sklearn for similarity computation

## ▶ How to Run

```bash
pip install -r requirements.txt
python run.py input/input.json input/ output.json
