import pytest
from rag_eval import (
    tokenize,
    bm25_retrieve,
    overlap_retrieve,
    retrieve,
    extractive_answer,
    token_f1,
    exact_match,
    run_eval,
    compare_configs,
)


@pytest.fixture
def corpus():
    return [
        {
            "id": "mercury",
            "text": "Mercury is the closest planet to the Sun. It is a small rocky world.",
        },
        {
            "id": "mars",
            "text": "Mars is known as the red planet. Its surface is covered with dust.",
        },
        {
            "id": "jupiter",
            "text": "Jupiter is the largest planet in the solar system. It has the Great Red Spot.",
        },
        {
            "id": "saturn",
            "text": "Saturn is famous for its rings. Its largest moon is Titan.",
        },
    ]


@pytest.fixture
def questions():
    return [
        {
            "id": "q1",
            "question": "Which planet is closest to the Sun?",
            "answer": "Mercury",
            "relevant": ["mercury"],
        },
        {
            "id": "q2",
            "question": "Which planet is known as the red planet?",
            "answer": "Mars",
            "relevant": ["mars"],
        },
        {
            "id": "q3",
            "question": "Which planet has rings and a moon named Titan?",
            "answer": "Saturn",
            "relevant": ["saturn"],
        },
    ]


def test_tokenize_lowercases_and_drops_stopwords():
    tokens = tokenize("The Quick brown Fox")
    assert "quick" in tokens
    assert "brown" in tokens
    assert "fox" in tokens
    assert "the" not in tokens


def test_tokenize_empty():
    assert tokenize("") == []


def test_bm25_ranks_relevant_first(corpus):
    results = bm25_retrieve("closest planet to the Sun", corpus, top_k=2)
    assert results[0]["doc"]["id"] == "mercury"


def test_bm25_returns_empty_when_no_match(corpus):
    assert bm25_retrieve("quantum banana", corpus, top_k=3) == []


def test_bm25_respects_top_k(corpus):
    results = bm25_retrieve("planet", corpus, top_k=1)
    assert len(results) <= 1


def test_overlap_baseline(corpus):
    results = overlap_retrieve("rings Titan", corpus, top_k=1)
    assert results[0]["doc"]["id"] == "saturn"


def test_retrieve_unknown_method_raises(corpus):
    with pytest.raises(ValueError):
        retrieve("planet", corpus, method="dense", top_k=2)


def test_extractive_answer_picks_sentence_with_question_terms(corpus):
    jupiter = next(doc for doc in corpus if doc["id"] == "jupiter")
    answer = extractive_answer(
        "Which planet is the largest?", retrieved=[{"doc": jupiter, "score": 1.0}]
    )
    assert "largest" in answer.lower()


def test_extractive_answer_prefers_rare_terms(corpus):
    # "Red Planet" must beat a shorter sentence that only shares the generic
    # words "known" and "planet" (regression: naive overlap picked Pluto).
    retrieved = [{"doc": doc, "score": 1.0} for doc in corpus]
    answer = extractive_answer(
        "Which planet is known as the red planet?",
        retrieved=retrieved,
        corpus=corpus,
    )
    assert "Mars" in answer
    assert "red planet" in answer.lower()


def test_extractive_answer_empty_retrieved():
    assert extractive_answer("anything", []) == ""


def test_token_f1_identical_is_one():
    assert token_f1("Mars is red", "Mars is red") == 1.0


def test_token_f1_disjoint_is_zero():
    assert token_f1("apple banana", "quantum rocket") == 0.0


def test_token_f1_partial_between_zero_and_one():
    score = token_f1("the red planet Mars", "Mars")
    assert 0.0 < score < 1.0


def test_exact_match_case_insensitive():
    assert exact_match("mars", "Mars") is True


def test_exact_match_substring():
    assert exact_match("The answer is Mars, the red planet", "Mars") is True


def test_exact_match_false():
    assert exact_match("Venus", "Mars") is False


def test_run_eval_shape(corpus, questions):
    result = run_eval(corpus, questions, method="bm25", top_k=2)
    assert "config" in result
    assert "aggregate" in result
    assert "per_question" in result

    agg = result["aggregate"]
    expected_keys = {
        "recall_at_k",
        "precision_at_k",
        "mrr",
        "mean_f1",
        "exact_match_rate",
        "avg_latency_ms",
        "n_questions",
    }
    assert expected_keys.issubset(agg.keys())
    assert agg["n_questions"] == 3
    assert len(result["per_question"]) == 3


def test_run_eval_perfect_retrieval_high_recall(corpus, questions):
    result = run_eval(corpus, questions, method="bm25", top_k=4)
    assert result["aggregate"]["recall_at_k"] > 0.8


def test_run_eval_empty_questions(corpus):
    result = run_eval(corpus, [], method="bm25", top_k=3)
    agg = result["aggregate"]
    assert agg["n_questions"] == 0
    assert agg["recall_at_k"] == 0.0


def test_compare_configs_returns_one_run_per_config(corpus, questions):
    configs = [
        {"method": "bm25", "top_k": 2},
        {"method": "overlap", "top_k": 2},
    ]
    results = compare_configs(corpus, questions, configs)
    assert len(results) == 2
    assert results[0]["config"]["method"] == "bm25"
    assert results[1]["config"]["method"] == "overlap"
