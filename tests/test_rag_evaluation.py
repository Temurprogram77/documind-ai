import pytest
from documents.services.rag import RAGService


# Golden evaluation dataset with questions, expected keywords, and irrelevant queries
GOLDEN_EVALUATION_DATA = [
    {
        "query": "What is the annual software budget?",
        "expected_substring": "$120,000",
        "should_find": True
    },
    {
        "query": "What is the policy for remote work stipends?",
        "expected_substring": "$500 per year",
        "should_find": True
    },
    {
        "query": "What is the deadline for Q3 deliverables?",
        "expected_substring": "September 30",
        "should_find": True
    },
    {
        "query": "What is the distance between Earth and Mars?",
        "expected_substring": None,
        "should_find": False  # Out of domain query -> should not find context
    },
    {
        "query": "Who is the prime minister of Australia?",
        "expected_substring": None,
        "should_find": False  # Out of domain query -> should not find context
    }
]


@pytest.fixture
def rag_service_instance():
    return RAGService()


@pytest.fixture
def indexed_test_chunks(rag_service_instance):
    """Pre-indexes sample company document chunks for user 999, doc 999."""
    user_id = 999
    doc_id = 999

    chunks = [
        {
            "id": f"eval_chunk_1_{user_id}_{doc_id}",
            "text": "The annual software budget for the engineering department is set at $120,000 for FY2026. All purchase requests must be approved by the department director.",
            "metadata": {
                "user_id": user_id,
                "document_id": doc_id,
                "filename": "company_policy.pdf",
                "page_number": 1,
                "chunk_index": 0,
            }
        },
        {
            "id": f"eval_chunk_2_{user_id}_{doc_id}",
            "text": "Employees who work remotely full-time are eligible for an equipment stipend of $500 per year. Receipts must be submitted by December 15.",
            "metadata": {
                "user_id": user_id,
                "document_id": doc_id,
                "filename": "company_policy.pdf",
                "page_number": 2,
                "chunk_index": 1,
            }
        },
        {
            "id": f"eval_chunk_3_{user_id}_{doc_id}",
            "text": "All Q3 deliverables and milestone milestones must be completed no later than September 30. Code reviews must conclude two days prior.",
            "metadata": {
                "user_id": user_id,
                "document_id": doc_id,
                "filename": "company_policy.pdf",
                "page_number": 3,
                "chunk_index": 2,
            }
        }
    ]

    rag_service_instance.ingest_chunks(chunks)
    return user_id, doc_id


class TestRAGEvaluation:

    def test_sentence_aware_chunking_preserves_sentences(self, rag_service_instance):
        """Validates that sentence boundary detection prevents cutting in the middle of words."""
        sample_text = (
            "First sentence about document intelligence. "
            "Second sentence explaining vector embeddings and similarity. "
            "Third sentence detailing multi-tenant ChromaDB isolation. "
            "Fourth sentence covering security and rate limiting."
        )

        boundary = rag_service_instance._find_sentence_boundary(sample_text, 45)
        # Should cut cleanly after a period or whitespace, not in the middle of a word
        cut_text = sample_text[:boundary].strip()
        assert cut_text.endswith(".") or cut_text.endswith("intelligence")

    def test_retrieval_hit_rate_on_golden_dataset(self, rag_service_instance, indexed_test_chunks):
        """Measures retrieval hit rate on in-domain questions (target >= 80%)."""
        user_id, doc_id = indexed_test_chunks
        hits = 0
        total_in_domain = 0

        for item in GOLDEN_EVALUATION_DATA:
            if not item["should_find"]:
                continue

            total_in_domain += 1
            contexts = rag_service_instance.query_context(
                user_id=user_id,
                document_id=doc_id,
                query=item["query"],
                top_k=2
            )

            # Check if expected substring is present in retrieved chunks
            found = any(item["expected_substring"] in c["text"] for c in contexts)
            if found:
                hits += 1

        hit_rate = hits / total_in_domain
        assert hit_rate >= 0.80, f"Retrieval hit-rate ({hit_rate:.2f}) is below 80% threshold."

    def test_citations_included_in_context(self, rag_service_instance, indexed_test_chunks):
        """Verifies that retrieved chunks retain page number metadata for citation."""
        user_id, doc_id = indexed_test_chunks
        contexts = rag_service_instance.query_context(
            user_id=user_id,
            document_id=doc_id,
            query="remote work equipment stipend",
            top_k=2
        )

        assert len(contexts) > 0
        top_result = contexts[0]
        assert "page_number" in top_result
        assert top_result["page_number"] == 2
