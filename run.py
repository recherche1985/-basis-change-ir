import json
from collections import defaultdict
import random

# Paths to your data files
QUERIES_FILE = "queries.json"   # should exist
QRELS_FILE = "qrels.txt"        # should exist
RESULTS_FILE = "run.json"       # will be generated

# Load queries
with open(QUERIES_FILE, "r", encoding="utf-8") as f:
    queries = json.load(f)  # {query_id: query_text}

# Load relevance judgments (qrels)
# Format: qrels[query_id] = set(doc_id)
qrels = defaultdict(set)
with open(QRELS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 3:
            qid, _, docid = parts[:3]
            qrels[qid].add(docid)

# Simulated retrieval function
def retrieve(query_text, top_k=100):
    """
    Simulate BM25 and BasisChange retrieval.
    Returns a list of dicts with doc_id, BM25 score, BasisChange score.
    """
    results = []
    for i in range(top_k):
        doc_id = f"D{i+1:05d}"  # e.g., D00001
        # Simulate scores
        bm25_score = random.uniform(0, 1)
        basis_score = bm25_score + random.uniform(-0.05, 0.1)  # Basis change tweak
        results.append({
            "doc_id": doc_id,
            "BM25": round(bm25_score, 4),
            "BasisChange": round(basis_score, 4)
        })
    # Sort by BM25 for baseline ranking
    results.sort(key=lambda x: x["BM25"], reverse=True)
    return results

# Run retrieval for all queries
all_results = []

for qid, query_text in queries.items():
    retrieved_docs = retrieve(query_text)
    for rank, doc in enumerate(retrieved_docs, start=1):
        all_results.append({
            "query_id": str(qid),
            "doc_id": doc["doc_id"],
            "rank": rank,
            "BM25": doc["BM25"],
            "BasisChange": doc["BasisChange"]
        })

# Save results to JSON
with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"[+] Retrieval results saved to {RESULTS_FILE}")
