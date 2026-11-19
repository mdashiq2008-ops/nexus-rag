# NEXUS RAG v1.0.0

## Highlights

- Local-first FastAPI and React application
- Multi-PDF upload, indexing, duplicate detection, library management and deletion
- Dense semantic, BM25 and hybrid retrieval
- Optional ONNX cross-encoder reranking
- Grounded extractive answers with verified citations
- Optional local llama.cpp / Qwen3 inference
- Clickable PDF page citations and expandable source passages
- Real system, database, document, reranker and local-model status

## Validation

- 59 Python tests passing
- Python compilation passing
- Frontend production build passing
- Frontend lint passing
- Clean Python virtual-environment dependency installation passing
- Clean frontend `npm ci` passing
- `npm audit`: zero vulnerabilities
- `pip-audit`: no known vulnerabilities
- Real Qwen Q4_K_M inference: passed
- Desktop and responsive browser interaction: passed

## Known limitations

- Real LLM performance depends heavily on available CPU and RAM.
- The measured Qwen startup and generation times are hardware-specific.
- Evidence relevance remains based on retrieval; the answer core validates exact
  excerpt selection but does not independently prove that every retrieved
  passage is relevant.
- OCR quality depends on the installed Tesseract runtime and language data.
