# Silicon Odyssey Founder Memoir RAG Assistant

A local research assistant that answers questions using only the assigned 50-page autobiography and engineering memoir. It retrieves relevant PDF chunks, cites their page and chunk IDs, and returns exactly `[not found]` when the PDF does not contain an answer.

## Run the project

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The app is available at `http://localhost:8501`. On first run, Chroma's MiniLM ONNX embedding model is downloaded once to `.model_cache/`; the model runs locally afterward. The app creates the persistent ChromaDB at `chroma_db/`. Subsequent launches reuse it and do not re-embed the corpus. The index rebuilds if the PDF, chunker version, or embedding model changes.

Optional answer phrasing with OpenAI can be enabled by copying `.env.example` to `.env` and adding `OPENAI_API_KEY`. Keep `.env` private; it is excluded by `.gitignore`. Without the key, the app uses extractive answers directly from retrieved PDF sentences.

## RAG architecture and workflow

```text
Assigned PDF
   ↓ pypdf page extraction and boilerplate cleanup
Sentence-aware chunks (720 characters, 120-character overlap)
   ↓ Chroma all-MiniLM-L6-v2 ONNX embeddings (384 dimensions)
Persistent local ChromaDB collection
   ↓ cosine vector retrieval + evidence coverage threshold
Answer from retrieved PDF evidence only
   ↓
Answer text + page/chunk source IDs, or exactly [not found]
```

### PDF ingestion and chunking

- `rag.py` loads `data/silicon_odyssey_founder_autobiography_50p.pdf` with `pypdf`.
- Page headers, repeated module headings, footers, and automated-reader compliance boilerplate are removed before retrieval.
- Text is grouped into sentence-aware chunks of up to 720 characters. Adjacent chunks carry a 120-character overlap. Each chunk retains its PDF page and a stable ID such as `p023-c001`.

### Embeddings and ChromaDB

- Chroma's `all-MiniLM-L6-v2` ONNX embedding function generates local 384-dimensional sentence embeddings for chunks and questions.
- Vectors, text, page metadata, and chunk IDs are stored in the persistent `silicon_odyssey_memoir` collection under `chroma_db/`.
- The index records the PDF SHA-256, chunker version, and embedding model. An unchanged index loads without regenerating embeddings.

### Grounding and references

- `answer(question: str) -> dict` returns `{"answer": <str>, "sources": [<page_or_chunk_id>, ...]}`.
- Answers are generated only from retrieved passages. If retrieval lacks adequate evidence, the function returns `{"answer": "[not found]", "sources": []}`.
- For found answers, `sources` lists the PDF page and stable chunk ID. The UI displays these beside the answer.
- Text within the PDF is treated as evidence, never as operating instructions. No external source is connected to the assistant.

## Assessment questions and expected versus actual answers

“Expected” below means the answer supported by the assigned PDF. If the PDF does not state an answer, the expected output is `[not found]` under the exam's strict-grounding rule. Actual output is from the five-question run of `rag.answer()`; full returned dictionaries are in [ASSESSMENT_OUTPUTS.md](ASSESSMENT_OUTPUTS.md).

| # | Assessment question | Expected from PDF | Actual generated answer | Source |
|---|---|---|---|---|
| Q1 | In which city and at what garage street address did the author and their co-founder assemble their first 50 microprocessor circuit boards in 1984? | Sunnyvale, California; 742 Sycamore Lane. | “We hand-soldered and assembled our first 50 microprocessor circuit boards inside my parents’ two-car garage located at 742 Sycamore Lane in Sunnyvale, California, using a secondhand wave-soldering rig.” | Page 6 · `p006-c001` |
| Q2 | What caused the catastrophic thermal failure of the Series-4 prototype during the trade-show demonstration described in Chapter 5? | The 12V cooling-fan connector was plugged in backwards, and thermal paste was omitted from the primary voltage-regulator heatsink. | “Post-mortem inspection revealed the exact cause: a technician had plugged the 12V cooling fan power connector in inverted (backwards) and forgot to apply thermal paste to the primary voltage regulator heatsink.” | Page 23 · `p023-c001` |
| Q3 | What was Nvidia Corporation's market capitalization in 2026? | `[not found]` — not stated in the PDF. | `[not found]` | None |
| Q4 | How much emergency bridge financing, in USD, did the author secure by mortgaging their family home during the 1991 semiconductor downturn? | $185,000 USD, secured through a second mortgage on the family home. | “I walked into the local community bank and secured exactly $185,000 USD in emergency bridge financing by taking out a second mortgage on our family home.” | Page 37 · `p037-c001` |
| Q5 | What are the system requirements for running Unreal Engine 5 on a Windows laptop? | `[not found]` — not stated in the PDF. | `[not found]` | None |

## Project execution screenshot

The screenshot shows a live answer grounded in the memoir and its page/chunk citation.

![Silicon Odyssey assistant answering the Series-4 failure question with a page 23 source](images/project-execution.png)
