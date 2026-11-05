import csv
import io
import json
import os

from flask import Flask, Response, jsonify, render_template, request
from rag_eval import compare_configs, load_corpus, load_questions, run_eval

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-me")


def _sample_path(name):
    return os.path.join(os.path.dirname(__file__), "data", name)


def _load_sample_corpus():
    try:
        return load_corpus(_sample_path("sample_corpus.json"))
    except Exception:
        return []


def _load_sample_questions():
    try:
        return load_questions(_sample_path("sample_questions.json"))
    except Exception:
        return []


def _parse_json_input(text, kind):
    try:
        items = json.loads(text)
    except json.JSONDecodeError as exc:
        return [], f"Invalid JSON for {kind}: {exc}"

    if not isinstance(items, list):
        return [], f"{kind} JSON must be a list"

    required = {
        "corpus": {"id", "title", "text"},
        "questions": {"id", "question", "answer", "relevant"},
    }[kind]

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            return [], f"{kind} item at index {idx} is not an object"
        missing = required - set(item.keys())
        if missing:
            return [], f"{kind} item at index {idx} missing key: {sorted(missing)[0]}"

    return items, None


def _parse_int(value, default, minimum, maximum):
    try:
        value = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


def _f1_svg(per_question):
    if not per_question:
        return ""

    rows = len(per_question)
    width = 560
    height = 22 * rows + 8
    bar_max = 420
    label_x = 40
    track_x = 50

    parts = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        '<style>text{font-family:sans-serif;font-size:12px;fill:#9aa7b8}</style>',
    ]

    for i, item in enumerate(per_question):
        y = 4 + i * 22
        f1 = float(item.get("f1", 0.0))
        f1 = max(0.0, min(1.0, f1))
        bar_width = round(f1 * bar_max)
        label = f"Q{i + 1}"

        parts.extend(
            [
                f'<text x="{label_x}" y="{y + 14}" text-anchor="end">{label}</text>',
                f'<rect x="{track_x}" y="{y + 4}" width="{bar_max}" height="12" fill="#2a3442" rx="2"/>',
                f'<rect x="{track_x}" y="{y + 4}" width="{bar_width}" height="12" fill="#e8b44a" rx="2"/>',
                f'<text x="{track_x + bar_max + 6}" y="{y + 14}">{f1:.2f}</text>',
            ]
        )

    parts.append("</svg>")
    return "".join(parts)


def build_eval(corpus, questions, top_k, method, compare):
    config = {"top_k": top_k, "method": method}
    result = run_eval(corpus, questions, top_k=top_k, method=method)

    comparison = None
    if compare:
        other_method = "overlap" if method == "bm25" else "bm25"
        comparison = compare_configs(
            corpus,
            questions,
            [
                {"top_k": top_k, "method": method},
                {"top_k": top_k, "method": other_method},
            ],
        )

    return {
        "result": result,
        "comparison": comparison,
        "n_docs": len(corpus),
        "n_questions": len(questions),
    }


@app.route("/", methods=["GET"])
def index():
    n_docs = len(_load_sample_corpus())
    n_questions = len(_load_sample_questions())
    return render_template("index.html", n_docs=n_docs, n_questions=n_questions, error=None)


@app.route("/evaluate", methods=["POST"])
def evaluate():
    corpus_source = request.form.get("corpus_source", "sample")
    questions_source = request.form.get("questions_source", "sample")
    top_k = _parse_int(request.form.get("top_k"), default=3, minimum=1, maximum=10)
    method = request.form.get("method", "bm25")
    if method not in ("bm25", "overlap"):
        method = "bm25"
    compare = request.form.get("compare") == "on"

    error = None

    if corpus_source == "sample":
        corpus = _load_sample_corpus()
        if not corpus:
            error = "Failed to load sample corpus."
    else:
        corpus, err = _parse_json_input(request.form.get("corpus_text", ""), "corpus")
        if err:
            error = err

    if not error:
        if questions_source == "sample":
            questions = _load_sample_questions()
            if not questions:
                error = "Failed to load sample questions."
        else:
            questions, err = _parse_json_input(
                request.form.get("questions_text", ""), "questions"
            )
            if err:
                error = err

    if not error and not corpus:
        error = "Corpus is empty."
    if not error and not questions:
        error = "Question set is empty."

    if error:
        return (
            render_template(
                "index.html",
                n_docs=len(_load_sample_corpus()),
                n_questions=len(_load_sample_questions()),
                error=error,
            ),
            200,
        )

    eval_data = build_eval(corpus, questions, top_k, method, compare)
    return render_template(
        "results.html",
        eval=eval_data,
        f1_svg=_f1_svg(eval_data["result"].get("per_question", [])),
        top_k=top_k,
        method=method,
        compare=compare,
    )


@app.route("/api/evaluate", methods=["GET"])
def api_evaluate():
    corpus = _load_sample_corpus()
    questions = _load_sample_questions()
    top_k = _parse_int(request.args.get("top_k"), default=3, minimum=1, maximum=10)
    method = request.args.get("method", "bm25")
    if method not in ("bm25", "overlap"):
        method = "bm25"
    compare = request.args.get("compare") in ("1", "true", "on", "yes")
    return jsonify(build_eval(corpus, questions, top_k, method, compare))


@app.route("/export.csv", methods=["GET"])
def export_csv():
    corpus = _load_sample_corpus()
    questions = _load_sample_questions()
    top_k = _parse_int(request.args.get("top_k"), default=3, minimum=1, maximum=10)
    method = request.args.get("method", "bm25")
    if method not in ("bm25", "overlap"):
        method = "bm25"

    result = run_eval(corpus, questions, top_k=top_k, method=method)
    per_question = result.get("per_question", [])

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "question_id",
            "question",
            "gold_answer",
            "predicted_answer",
            "f1",
            "exact_match",
            "hit",
            "recall",
            "precision",
            "retrieved_ids",
            "latency_ms",
        ]
    )

    for item in per_question:
        retrieved = item.get("retrieved_ids", [])
        if not isinstance(retrieved, (list, tuple)):
            retrieved = [retrieved]
        writer.writerow(
            [
                item.get("id", ""),
                item.get("question", ""),
                item.get("gold_answer", ""),
                item.get("predicted_answer", ""),
                item.get("f1", 0.0),
                item.get("exact", False),
                item.get("hit", False),
                item.get("recall", 0.0),
                item.get("precision", 0.0),
                ";".join(str(r) for r in retrieved),
                item.get("latency_ms", 0.0),
            ]
        )

    csv_text = output.getvalue()
    output.close()

    headers = {"Content-Disposition": "attachment; filename=rag_eval_results.csv"}
    return Response(csv_text, mimetype="text/csv", headers=headers)


if __name__ == "__main__":
    app.run(debug=True)
