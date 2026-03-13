"""Sentinel Co-Pilot retrieval agent for active-policy Q&A.

This script enforces strict graph retrieval for Tier-1 banking governance:
- Only active policies are retrieved (no incoming SUPERSEDES edge).
- Answers are synthesized from retrieved active context only.
- If no verified active context exists, returns a strict fallback response.
"""

from __future__ import annotations

import os
from typing import List, Sequence

from dotenv import find_dotenv, load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from neo4j import Driver, GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable
from pydantic import BaseModel, Field

STRICT_NO_ANSWER = (
    "I cannot find a verified active policy for this in the current database."
)


class ActivePolicy(BaseModel):
    """Verified active policy context returned from Neo4j retrieval."""

    document_name: str = Field(..., description="Policy document identifier.")
    category: str = Field(..., description="SME-governed ontology category.")
    customer_types: List[str] = Field(
        default_factory=list,
        description="Customer types explicitly connected through APPLIES_TO edges.",
    )
    required_docs: List[str] = Field(
        default_factory=list,
        description="Required documents explicitly connected through REQUIRES edges.",
    )
    extracted_rule: str = Field(..., description="Normalized policy rule summary.")
    source_text: str = Field(..., description="Original policy source text.")
    score: float = Field(..., description="Vector similarity score from Neo4j index.")


def load_environment() -> None:
    """Load .env and normalize quoted credential fields."""

    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path=dotenv_path, override=True)

    # Normalize values to avoid quoted strings breaking auth/uri parsing.
    for key in ("GROQ_API_KEY", "NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"):
        value = os.environ.get(key)
        if value is not None:
            os.environ[key] = value.strip().strip('"').strip("'")


def _load_and_sanitize_env() -> None:
    """Backward-compatible alias for older callers."""

    load_environment()


def _to_bolt_uri(uri: str) -> str:
    """Convert routing URI forms to direct bolt URI for single-node local setups."""

    if uri.startswith("neo4j://"):
        return uri.replace("neo4j://", "bolt://", 1)
    if uri.startswith("neo4j+s://"):
        return uri.replace("neo4j+s://", "bolt+s://", 1)
    if uri.startswith("neo4j+ssc://"):
        return uri.replace("neo4j+ssc://", "bolt+ssc://", 1)
    return uri



def build_neo4j_driver() -> Driver:
    """Create Neo4j driver and fallback to direct bolt when routing is unavailable."""

    uri = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")

    # Local single-node Neo4j typically does not expose routing tables.
    # Convert local neo4j:// URIs to bolt:// before creating the driver.
    if uri.startswith(("neo4j://", "neo4j+s://", "neo4j+ssc://")) and (
        "127.0.0.1" in uri or "localhost" in uri
    ):
        direct_uri = _to_bolt_uri(uri)
        print(f"Using direct local Neo4j URI: {direct_uri}")
        uri = direct_uri

    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        print(f"Neo4j connected using URI: {uri}")
        return driver
    except ServiceUnavailable as error:
        error_text = str(error).lower()
        if "routing" not in error_text:
            raise

        fallback_uri = _to_bolt_uri(uri)
        if fallback_uri == uri:
            raise

        print(
            "Routing unavailable on current Neo4j endpoint; "
            f"retrying with direct URI: {fallback_uri}"
        )
        driver = GraphDatabase.driver(fallback_uri, auth=(user, password))
        driver.verify_connectivity()
        os.environ["NEO4J_URI"] = fallback_uri
        return driver


def build_groq_llm() -> ChatGroq:
    """Create Groq LLM client used for response synthesis."""

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set. Export it before running this script.")

    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=api_key)


def build_embeddings_model() -> HuggingFaceEmbeddings:
    """Create local sentence-transformer embeddings model for semantic retrieval."""

    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def retrieve_active_policy(
    driver: Driver,
    user_question: str,
    question_embedding: Sequence[float],
    top_k: int = 5,
) -> List[ActivePolicy]:
    """Retrieve active policies with vector search and governance filtering.

    Business logic:
    - Semantic candidate generation via Neo4j vector index on Policy.embedding.
    - Strict governance filter is enforced with `WHERE NOT ()-[:SUPERSEDES]->(p)`.
    - Returns evidence-ready context (rule, document name, category).
    """

    cypher_query = """
    CALL {
        WITH $question_embedding AS qe, $top_k AS tk
        CALL db.index.vector.queryNodes('policy_embeddings', tk, qe)
        YIELD node AS p, score AS vector_score
        RETURN p, vector_score, 0.0 AS text_score

        UNION ALL

        WITH $user_question AS uq, $top_k AS tk
        CALL db.index.fulltext.queryNodes('policy_keywords', uq, {limit: tk})
        YIELD node AS p, score AS raw_text_score
        RETURN p, 0.0 AS vector_score, raw_text_score AS text_score
    }
    // 1. Score Fusion
    WITH p, max(vector_score) AS vs, max(text_score) AS ts
    // Normalize BM25 score roughly and combine with vector score
    WITH p, (vs + (ts / 10.0)) AS combined_score

    // 2. Governance Firewall
    MATCH (p)-[:BELONGS_TO]->(c:Category)
    WHERE NOT ()-[:SUPERSEDES]->(p)

    // 3. Multi-Hop Extraction
    OPTIONAL MATCH (p)-[:APPLIES_TO]->(ct:CustomerType)
    OPTIONAL MATCH (p)-[:REQUIRES]->(dr:DocumentRequirement)
    WITH p, c, combined_score, collect(DISTINCT ct.name) AS customer_types, collect(DISTINCT dr.name) AS required_docs
    RETURN p.name AS document_name,
           c.name AS category,
           coalesce(p.extracted_rule, "") AS extracted_rule,
           coalesce(p.source_text, "") AS source_text,
           customer_types,
           required_docs,
           combined_score AS score
    ORDER BY score DESC
    LIMIT $top_k
    """

    vector_only_query = """
    CALL db.index.vector.queryNodes('policy_embeddings', $top_k, $question_embedding)
    YIELD node AS p, score
    MATCH (p)-[:BELONGS_TO]->(c:Category)
    WHERE NOT ()-[:SUPERSEDES]->(p)
    OPTIONAL MATCH (p)-[:APPLIES_TO]->(ct:CustomerType)
    OPTIONAL MATCH (p)-[:REQUIRES]->(dr:DocumentRequirement)
    WITH p, c, score, collect(DISTINCT ct.name) AS customer_types, collect(DISTINCT dr.name) AS required_docs
    RETURN p.name AS document_name,
           c.name AS category,
           coalesce(p.extracted_rule, "") AS extracted_rule,
           coalesce(p.source_text, "") AS source_text,
           customer_types,
           required_docs,
           score
    ORDER BY score DESC
    LIMIT $top_k
    """

    def _is_missing_fulltext_index(error: Neo4jError) -> bool:
        """Detect missing full-text index errors for graceful hybrid fallback."""

        error_text = str(error).lower()
        return (
            "policy_keywords" in error_text
            and "index" in error_text
            and (
                "does not exist" in error_text
                or "not found" in error_text
                or "unknown" in error_text
                or "there is no such" in error_text
            )
        )

    try:
        with driver.session() as session:
            query_params = {
                "user_question": user_question,
                "question_embedding": [float(value) for value in question_embedding],
                "top_k": top_k,
            }
            try:
                records = session.execute_read(
                    lambda tx: list(
                        tx.run(
                            cypher_query,
                            **query_params,
                        )
                    )
                )
            except Neo4jError as error:
                if not _is_missing_fulltext_index(error):
                    raise
                print(
                    "Full-text index 'policy_keywords' is unavailable; "
                    "falling back to vector-only retrieval."
                )
                records = session.execute_read(
                    lambda tx: list(
                        tx.run(
                            vector_only_query,
                            question_embedding=query_params["question_embedding"],
                            top_k=top_k,
                        )
                    )
                )

        return [
            ActivePolicy(
                document_name=record["document_name"],
                category=record["category"],
                customer_types=[
                    value
                    for value in (record.get("customer_types") or [])
                    if value is not None
                ],
                required_docs=[
                    value
                    for value in (record.get("required_docs") or [])
                    if value is not None
                ],
                extracted_rule=record["extracted_rule"],
                source_text=record["source_text"],
                score=float(record["score"]),
            )
            for record in records
        ]
    except ServiceUnavailable as error:
        print(f"Neo4j connection dropped during retrieval: {error}")
        return []
    except Neo4jError as error:
        print(f"Neo4j query error during retrieval: {error}")
        return []
    except Exception as error:
        print(f"Unexpected retrieval error: {error}")
        return []


def generate_answer(
    llm: ChatGroq,
    active_context: Sequence[ActivePolicy],
    user_question: str,
) -> str:
    """Generate a grounded answer using only verified active policy context."""

    if not active_context:
        return STRICT_NO_ANSWER

    context_blocks = []
    for item in active_context:
        context_blocks.append(
            (
                f"Document: {item.document_name}\n"
                f"Category: {item.category}\n"
                f"Applies To: {', '.join(item.customer_types) if item.customer_types else 'None'}\n"
                f"Requires: {', '.join(item.required_docs) if item.required_docs else 'None'}\n"
                f"Rule: {item.extracted_rule}"
            )
        )
    context_text = "\n\n".join(context_blocks)

    prompt = PromptTemplate.from_template(
        """
You are the Sentinel Banking Co-Pilot.
Answer the user's question using ONLY the provided active_context.
If the context is empty or does not contain the answer, you MUST reply with:
"I cannot find a verified active policy for this in the current database."
Always cite the document name.

active_context:
{active_context}

user_question:
{user_question}
""".strip()
    )

    try:
        formatted_prompt = prompt.format(
            active_context=context_text,
            user_question=user_question,
        )
        response = llm.invoke(formatted_prompt)
        return str(response.content).strip()
    except Exception as error:
        error_text = str(error)
        if "429" in error_text or "rate" in error_text.lower():
            return (
                "Groq API rate limit encountered while generating response. "
                "Please retry in a few seconds."
            )
        return f"Failed to generate response from Groq: {error}"


def print_response(answer: str, active_context: Sequence[ActivePolicy]) -> None:
    """Print user-friendly answer plus evidence snapshot."""

    if active_context:
        evidence = ", ".join(
            f"{item.document_name} [{item.category}] (score={item.score:.4f})"
            for item in active_context
        )
    else:
        evidence = "None"

    print("\nAnswer:")
    print(answer)
    print(f"Source: {evidence}\n")


def main() -> None:
    """Run interactive retrieval loop for Sentinel Co-Pilot."""

    load_environment()

    try:
        driver = build_neo4j_driver()
    except ServiceUnavailable as error:
        print("Neo4j is not reachable. Start Neo4j and confirm Bolt is enabled.")
        print(
            "Expected endpoint: "
            f"{os.getenv('NEO4J_URI', 'bolt://127.0.0.1:7687')}"
        )
        print(f"Connection error: {error}")
        return
    except Neo4jError as error:
        print(f"Neo4j startup check failed: {error}")
        return

    try:
        llm = build_groq_llm()
        embeddings_model = build_embeddings_model()
        print("Sentinel Co-Pilot is ready. Type 'exit' to quit.")

        while True:
            user_question = input("\nAsk Sentinel> ").strip()
            if user_question.lower() in {"exit", "quit", "q"}:
                print("Exiting Sentinel Co-Pilot.")
                break

            question_embedding = embeddings_model.embed_query(user_question)
            active_context = retrieve_active_policy(
                driver,
                user_question,
                question_embedding,
                top_k=5,
            )
            answer = generate_answer(llm, active_context, user_question)
            print_response(answer, active_context)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
