import json
import math
import re
import time

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "and", "but", "or", "yet",
    "so", "if", "because", "while", "where", "when", "that", "which",
    "who", "what", "this", "these", "those", "it", "its", "they", "them",
    "their", "we", "our", "us", "i", "my", "me", "you", "your"
}

_WORD_RE = re.compile(r"[a-z0-9]+")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_NORM_RE = re.compile(r"[^a-z0-9]+")


def tokenize(text):
    tokens = _WORD_RE.findall(str(text).lower())
    return [t for t in tokens if len(t) >= 2 and t not in STOPWORDS]


def load_corpus(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_questions(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def bm25_retrieve(query, docs, top_k, k1=1.5, b=0.75):
    query_terms = set(tokenize(query))
    if not query_terms or not docs:
        return []

    doc_tokens = [tokenize(doc.get("text", "")) for doc in docs]
    doc_lengths = [len(toks) for toks in doc_tokens]
    avgdl = sum(doc_lengths) / len(doc_lengths)

    N = len(docs)
    df = {}
    for term in query_terms:
        df[term] = sum(1 for toks in doc_tokens if term in toks)

    scored = []
    for doc, toks, length in zip(docs, doc_tokens, doc_lengths):
        score = 0.0
        for term in query_terms:
            f = toks.count(term)
            if f == 0:
                continue
            idf = math.log(1 + (N - df[term] + 0.5) / (df[term] + 0.5))
            denom = f + k1 * (1 - b + b * (length / avgdl))
            score += idf * f * (k1 + 1) / denom
        if score > 0:
            scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"doc": doc, "score": float(score)} for score, doc in scored[:top_k]]


def overlap_retrieve(query, docs, top_k):
    query_terms = set(tokenize(query))
    if not query_terms or not docs:
        return []

    scored = []
    for doc in docs:
        doc_terms = set(tokenize(doc.get("text", "")))
        overlap = len(query_terms & doc_terms)
        if overlap > 0:
            scored.append((overlap, doc.get("id", ""), doc))

    scored.sort(key=lambda x: (-x[0], x[1]))
    return [{"doc": doc, "score": float(overlap)} for overlap, _, doc in scored[:top_k]]


def retrieve(query, docs, top_k, method):
    if method == "bm25":
        return bm25_retrieve(query, docs, top_k)
    if method == "overlap":
        return overlap_retrieve(query, docs, top_k)
    raise ValueError(f"Unknown retrieval method: {method}")


def extractive_answer(question, retrieved, corpus=None):
    if not retrieved:
        return ""

    q_terms = set(tokenize(question))
    if not q_terms:
        return ""

    # Weight question terms by rarity, so generic words ("planet", "solar
    # system") do not outvote discriminative ones ("hottest", "red") when
    # several sentences match. Rarity is measured over the full corpus when
    # given, else over the retrieved docs.
    if corpus:
        idf_docs = [doc.get("text", "") for doc in corpus]
    else:
        idf_docs = [item.get("doc", {}).get("text", "") for item in retrieved]
    n_docs = len(idf_docs)
    weights = {}
    for term in q_terms:
        df = sum(1 for t in idf_docs if term in set(tokenize(t)))
        weights[term] = math.log(1 + n_docs / max(1, df))

    best_sentence = ""
    best_score = -1.0
    best_len = float("inf")

    for item in retrieved:
        text = item.get("doc", {}).get("text", "")
        for sent in _SENT_RE.split(text):
            sent = sent.strip()
            if not sent:
                continue
            matched = set(tokenize(sent)) & q_terms
            score = sum(weights[t] for t in matched)
            sent_len = len(sent)
            if score > best_score or (score == best_score and sent_len < best_len):
                best_sentence = sent
                best_score = score
                best_len = sent_len

    return best_sentence


def token_f1(prediction, gold):
    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counts = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    gold_counts = {}
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1

    tp = 0
    for token, count in pred_counts.items():
        tp += min(count, gold_counts.get(token, 0))

    if tp == 0:
        return 0.0

    precision = tp / len(pred_tokens)
    recall = tp / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction, gold):
    pred_norm = _NORM_RE.sub(" ", str(prediction).lower()).strip()
    gold_norm = _NORM_RE.sub(" ", str(gold).lower()).strip()
    return pred_norm == gold_norm or gold_norm in pred_norm


def run_eval(corpus, questions, top_k=3, method="bm25"):
    per_question = []
    recalls = []
    precisions = []
    mrrs = []
    f1s = []
    exacts = []
    latencies = []

    for q in questions:
        start = time.perf_counter()
        retrieved = retrieve(q["question"], corpus, top_k, method)
        predicted = extractive_answer(q["question"], retrieved, corpus)
        latency = round((time.perf_counter() - start) * 1000, 1)

        retrieved_ids = [r["doc"]["id"] for r in retrieved]
        retrieved_titles = [r["doc"].get("title", "") for r in retrieved]
        relevant_ids = q.get("relevant", [])
        relevant_set = set(relevant_ids)

        hits = [rid for rid in retrieved_ids if rid in relevant_set]
        recall = len(hits) / len(relevant_ids) if relevant_ids else 0.0
        precision = len(hits) / len(retrieved_ids) if retrieved_ids else 0.0
        hit = len(hits) > 0

        mrr = 0.0
        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in relevant_set:
                mrr = 1.0 / rank
                break

        f1 = token_f1(predicted, q["answer"])
        exact = exact_match(predicted, q["answer"])

        recalls.append(recall)
        precisions.append(precision)
        mrrs.append(mrr)
        f1s.append(f1)
        exacts.append(1.0 if exact else 0.0)
        latencies.append(latency)

        per_question.append({
            "id": q["id"],
            "question": q["question"],
            "gold_answer": q["answer"],
            "predicted_answer": predicted,
            "f1": round(f1, 3),
            "exact": exact,
            "retrieved_ids": retrieved_ids,
            "retrieved_titles": retrieved_titles,
            "relevant_ids": relevant_ids,
            "hit": hit,
            "recall": round(recall, 3),
            "precision": round(precision, 3),
            "latency_ms": latency
        })

    n = len(questions)
    if n == 0:
        aggregate = {
            "recall_at_k": 0.0,
            "precision_at_k": 0.0,
            "mrr": 0.0,
            "mean_f1": 0.0,
            "exact_match_rate": 0.0,
            "avg_latency_ms": 0.0,
            "n_questions": 0
        }
    else:
        aggregate = {
            "recall_at_k": round(sum(recalls) / n, 3),
            "precision_at_k": round(sum(precisions) / n, 3),
            "mrr": round(sum(mrrs) / n, 3),
            "mean_f1": round(sum(f1s) / n, 3),
            "exact_match_rate": round(sum(exacts) / n, 3),
            "avg_latency_ms": round(sum(latencies) / n, 1),
            "n_questions": n
        }

    return {
        "config": {"top_k": top_k, "method": method},
        "aggregate": aggregate,
        "per_question": per_question
    }


def compare_configs(corpus, questions, configs):
    return [run_eval(corpus, questions, cfg["top_k"], cfg["method"]) for cfg in configs]
