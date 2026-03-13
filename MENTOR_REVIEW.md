# Sentinel GraphRAG - Mentor Review Notes

## 1. Project Objective
Sentinel is a banking governance GraphRAG prototype with two agentic pipelines:
- Curator Agent (ingestion): parse policy documents, map to ontology, and update supersession chains.
- Co-Pilot Agent (retrieval): answer user questions only from active policy context with evidence citation.

This PoC demonstrates strict governance behavior:
- superseded policies are excluded from answers,
- user responses are grounded in graph evidence,
- fallback response is returned when active evidence is missing.

---

## 2. Tech Stack Implemented
- Python 3.10+
- Neo4j (graph + vector index)
- LangChain + ChatGroq (`llama-3.3-70b-versatile`)
- HuggingFace sentence embeddings (`all-MiniLM-L6-v2`)
- Pydantic schemas for structured extraction
- Streamlit UI for module navigation and Co-Pilot chat

---

## 3. What We Built

### 3.1 `init_graph.py` (Graph Initialization + Ingestion)
Implemented:
- Ontology bootstrap (`Category` nodes):
  - `Retail_Loans`, `Corporate_Banking`, `KYC_AML`, `Credit_Cards`, `Tax_Compliance`
- Synthetic policy corpus (banking realistic examples):
  - AML 2024 PAN threshold (`INR 100,000`)
  - AML 2026 urgent update (`INR 50,000`) replacing 2024 memo
  - Retail NRI home loan policy (`8.35%`, up to `INR 20,000,000`)
- Curator schema (`GraphAction`) with strict fields:
  - `target_node`, `action_type`, `extracted_rule`, `superseded_document`
- Structured extraction via `with_structured_output(GraphAction)`
- Policy ingestion into Neo4j:
  - `(:Policy)-[:BELONGS_TO]->(:Category)`
  - supersession link: `(:Policy)-[:SUPERSEDES]->(:Policy)`
  - superseded node marked inactive (`active=false`)
- Vector retrieval support:
  - generated embeddings per policy
  - stored in `Policy.embedding`
  - vector index `policy_embeddings` created in Neo4j
- Robust error handling:
  - Neo4j connection/service errors
  - Neo4j query errors
  - LLM schema validation failures

### 3.2 `query_copilot.py` (Retrieval Agent)
Implemented:
- Environment normalization and `.env` loading
- Neo4j connection hardening:
  - local `neo4j://` auto-converted to `bolt://` for single-node setup
  - connectivity verification before runtime loop
- Vector retrieval query:
  - `db.index.vector.queryNodes('policy_embeddings', ...)`
  - strict active-policy filter: `WHERE NOT ()-[:SUPERSEDES]->(p)`
- LLM answer synthesis from retrieved context only
- strict fallback:
  - `I cannot find a verified active policy for this in the current database.`
- terminal interactive Q&A mode

### 3.3 `app.py` (Streamlit Integration)
Implemented:
- Sidebar navigation modules:
  - Dashboard
  - Curator Agent
  - Universal Ingestion
  - `💬 Co-Pilot (Retrieval)`
- Co-Pilot chat using:
  - `st.chat_input`
  - `st.chat_message`
- Cached resources to reduce lag:
  - Neo4j driver
  - Groq LLM
  - embeddings model
- UI evidence section with source citation expander
- relevance filtering before citation display to reduce noisy sources
- clear chat button and graceful error handling

---

## 4. Key Problems Encountered and How We Solved Them

### Issue A: "Unable to retrieve routing information"
Cause:
- Local Neo4j endpoint used routing URI (`neo4j://`) in a non-cluster setup.

Fix:
- Added local URI conversion to `bolt://` in retrieval driver builder.
- Added connectivity verification and better startup messages.

### Issue B: Connection refused (`WinError 10061`)
Cause:
- Neo4j service not listening on `127.0.0.1:7687` at that moment.

Fix:
- Added clear diagnostics and user-friendly startup failure guidance.
- Verified port/service externally before continuing.

### Issue C: Neo4j warning for unknown property `p.text`
Cause:
- Query referenced non-existent property name.

Fix:
- Standardized on `p.source_text` and removed `p.text` fallback in strict query path.

### Issue D: Source citation showing unrelated documents
Cause:
- Initial citation used full retrieved set or permissive lexical checks.

Fix:
- Added question-term filtering and overlap scoring in `app.py`.
- Removed generic stopwords and kept best-matching context for citation.

---

## 5. Current End-to-End Flow
1. Run ingestion (`init_graph.py`) to reset ontology and ingest policy docs.
2. Curator extraction maps each doc to ontology + supersession action.
3. Graph stores policy nodes, category edges, supersession lineage, embeddings.
4. Co-Pilot receives question and generates embedding.
5. Neo4j vector index retrieves semantic candidates.
6. Strict filter removes superseded policies.
7. LLM answers only from filtered active context.
8. Streamlit displays answer + source citation.

---

## 6. Demo Script for Mentor Review
Use these questions in Streamlit Co-Pilot:
1. `What is the PAN card limit for cash deposits?`
Expected:
- Answer from 2026 AML urgent circular
- PAN threshold should be INR 50,000
- Citation should point to AML circular category `KYC_AML`

2. `What is the interest rate for NRI home loans?`
Expected:
- Answer from Retail NRI home loan policy
- Interest 8.35% p.a.
- Citation should point to Retail Loans policy

3. `What is the card fee waiver in this database?`
Expected:
- strict fallback response if no active verified policy exists

---

## 7. How to Run
```powershell
# 1) install deps in venv
pip install -r requirements.txt

# 2) ingest graph data
python init_graph.py

# 3) run Streamlit app
streamlit run app.py
```

---

## 8. Current Limitations
- Dataset is still synthetic and small (3 docs).
- Citation filtering in app layer is lexical + overlap based; stronger reranking can be added.
- Multi-policy conflict resolution can be improved with explicit confidence thresholds.

---

## 9. Recommended Next Steps
1. Add policy version metadata (`effective_from`, `effective_to`, `regulator_ref`) to strengthen auditability.
2. Add explicit retrieval confidence threshold; return fallback if below threshold.
3. Add unit/integration tests for:
   - supersession chain behavior,
   - active-only retrieval constraint,
   - citation precision.
4. Add ingestion pipeline for real PDFs with chunk-level provenance references.
5. Introduce role-based access and policy domain restrictions for enterprise deployment.

---

## 10. Deliverables Created
- `init_graph.py`
- `query_copilot.py`
- `app.py`
- `requirements.txt`
- `MENTOR_REVIEW.md` (this document)
