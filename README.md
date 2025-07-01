# AI Research Paper Co-Author

A modern Streamlit app that acts as a true AI research paper co-author, inspired by the QuantaQuill neurosymbolic multi-agent framework.

## 🚀 Features Implemented

- **Multi-Agent Architecture**
  - Modular agents for research extraction, writing, and citation.
  - Each agent's output is visually distinct and conversational.

- **Real Source Integration**
  - Fetches real research papers from Semantic Scholar using extracted keywords.
  - Summarizes abstracts and generates contextual, human-like draft paragraphs.
  - Citations are formatted in academic style and reference real, up-to-date sources.

- **Knowledge Graph (Preview)**
  - Tracks extracted research elements, sources, and draft content.
  - Displays a JSON preview of the knowledge graph for transparency.

- **Modern UI/UX**
  - Streamlit app with tabbed workflow (Prompt Extraction, Drafting).
  - Chat-style drafting flow with agent messages and editable draft paragraphs.
  - Expanders and visual hierarchy for a clean, modular experience.

- **Smarter Drafting**
  - Bullet-point summaries from real abstracts.
  - Draft paragraphs synthesize and contextualize cited work, avoiding redundancy.
  - Human-like, QuantaQuill-inspired writing style.

## 🛠️ How It Works

1. **Prompt Extraction**: Enter your research idea in natural language. The system extracts domain, keywords, methods, objectives, and key concepts.
2. **Drafting (AI Co-Author)**: The app fetches real papers, summarizes key findings, and drafts a contextual paragraph with proper citations.
3. **Knowledge Graph**: All extracted elements and sources are tracked and previewed for transparency.

## 📦 Tech Stack
- Python 3
- Streamlit
- Semantic Scholar API (for real research sources)
- Pandas

## 📝 Example Use Cases
- Drafting the introduction for a research paper with real, cited sources.
- Exploring how multi-agent AI can assist in academic writing.
- Demonstrating neurosymbolic, explainable AI for scientific communication.

## ⚡ Current Status
- Core multi-agent, real-source, and drafting logic complete.
- Modern, modular UI/UX.
- Ready for further extension (validation agent, advanced knowledge graph, etc.).

---

*Inspired by the QuantaQuill framework for neurosymbolic, multi-agent scientific writing.*
