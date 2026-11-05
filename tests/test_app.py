import pytest
from app import app, _f1_svg


@pytest.fixture(scope="module")
def client():
    app.testing = True
    with app.test_client() as client:
        yield client


def test_home_loads(client):
    rv = client.get("/")
    assert rv.status_code == 200
    text = rv.get_data(as_text=True)
    assert "RAG Eval Harness" in text
    assert "Run evaluation" in text


def test_evaluate_sample_bm25(client):
    rv = client.post(
        "/evaluate",
        data={
            "corpus_source": "sample",
            "questions_source": "sample",
            "top_k": "3",
            "method": "bm25",
        },
    )
    assert rv.status_code == 200
    text = rv.get_data(as_text=True)
    assert "Recall@k" in text
    assert "Per-question breakdown" in text


def test_evaluate_with_compare(client):
    rv = client.post(
        "/evaluate",
        data={
            "corpus_source": "sample",
            "questions_source": "sample",
            "top_k": "3",
            "method": "bm25",
            "compare": "on",
        },
    )
    assert rv.status_code == 200
    assert "Head-to-head comparison" in rv.get_data(as_text=True)


def test_evaluate_overlap_method(client):
    rv = client.post(
        "/evaluate",
        data={
            "corpus_source": "sample",
            "questions_source": "sample",
            "top_k": "3",
            "method": "overlap",
        },
    )
    assert rv.status_code == 200
    assert "overlap" in rv.get_data(as_text=True).lower()


def test_evaluate_bad_pasted_json_shows_error(client):
    rv = client.post(
        "/evaluate",
        data={
            "corpus_source": "paste",
            "corpus_text": "{not json",
            "questions_source": "sample",
            "top_k": "3",
            "method": "bm25",
        },
    )
    assert rv.status_code == 200
    text = rv.get_data(as_text=True).lower()
    assert "error" in text or "invalid json" in text


def test_evaluate_empty_pasted_corpus_shows_error(client):
    rv = client.post(
        "/evaluate",
        data={
            "corpus_source": "paste",
            "corpus_text": "[]",
            "questions_source": "sample",
            "top_k": "3",
            "method": "bm25",
        },
    )
    assert rv.status_code == 200
    text = rv.get_data(as_text=True).lower()
    assert "error" in text or "empty" in text


def test_api_evaluate_json(client):
    rv = client.get("/api/evaluate?top_k=3&method=bm25")
    assert rv.status_code == 200
    data = rv.get_json()
    assert {"result", "comparison", "n_docs", "n_questions"} <= set(data.keys())
    recall = data["result"]["aggregate"]["recall_at_k"]
    assert isinstance(recall, float)
    assert 0.0 <= recall <= 1.0
    assert data["n_questions"] == 10


def test_api_evaluate_compare_off(client):
    rv = client.get("/api/evaluate?top_k=3&method=bm25")
    assert rv.status_code == 200
    assert rv.get_json()["comparison"] is None


def test_export_csv(client):
    rv = client.get("/export.csv?top_k=3&method=bm25")
    assert rv.status_code == 200
    assert "text/csv" in rv.content_type
    body = rv.get_data(as_text=True)
    assert body.startswith("question_id,question,gold_answer")
    lines = body.strip().splitlines()
    assert len(lines) == 11


def test_f1_svg_helper():
    assert _f1_svg([]) == ""
    svg = _f1_svg([{"id": "q1", "f1": 0.5}])
    assert "<svg" in svg
    assert "0.5" in svg
