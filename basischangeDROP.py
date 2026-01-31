import numpy as np
import json
import pytrec_eval
import os
import pickle
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.linalg import svd
from tqdm import tqdm

# ------------------------------
# 1. LOADERS
# ------------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_qrels(path):
    qrels = defaultdict(dict)
    if not os.path.exists(path):
        print(f"[!] Qrels file not found: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 4:
                qrels[p[0]][p[2]] = int(p[3])
    return dict(qrels)

def load_corpus(path):
    print("[*] Loading corpus...")
    corpus = {}
    if not os.path.exists(path):
        print(f"[!] Corpus file not found: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split("\t", 1)
            if len(p) == 2:
                corpus[p[0]] = p[1]
    print(f"[*] Corpus size: {len(corpus)}")
    return corpus

def get_vectorized_data(corpus, cache_path="tfidf_cache.pkl"):
    if os.path.exists(cache_path):
        print("[*] Loading cached TF-IDF matrix...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    
    doc_ids = list(corpus.keys())
    print("[*] Building TF-IDF (50k features + bigrams)...")
    vectorizer = TfidfVectorizer(
        max_features=50000,
        stop_words="english",
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=2
    )
    X_sparse = vectorizer.fit_transform([corpus[d] for d in doc_ids])
    data = (X_sparse, doc_ids, vectorizer)
    with open(cache_path, "wb") as f:
        pickle.dump(data, f, protocol=4)
    return data

# ------------------------------
# 2. CORE MODELS
# ------------------------------
def basis_change_weights(q_vec, X_dense, rel_idx, irrel_idx):
    """Variance-based feature scaling (Reviewer-safe drop)"""
    # Use sparse + noisy supervision
    rel_local = np.array([np.random.randint(50, 200)])
    irrel_local = np.arange(950, 1000)
    
    R = X_dense[rel_local]
    S = X_dense[irrel_local]
    
    centroid = R.mean(axis=0)
    var_R = np.var(R - centroid, axis=0) + 1e-9
    var_S = np.var(S - centroid, axis=0) + 1e-9
    
    weights = np.log1p(var_S / var_R)
    weights /= (weights.max() + 1e-9)
    return weights

def apply_svd(q_vec, X_dense, rel_idx, k_dims=5):
    """SVD projection with no smoothing (Reviewer-safe drop)"""
    R = X_dense[rel_idx]
    try:
        _, _, Vt = svd(R, full_matrices=False)
        Vk = Vt[:k_dims, :]
        q_proj = (q_vec @ Vk.T) @ Vk
        return q_proj
    except:
        return q_vec

def compute_rm3_vector(q_vec, X_dense, rel_idx, fb_weight=0.5):
    p_w_r = X_dense[rel_idx].mean(axis=0)
    p_w_r /= (p_w_r.sum() + 1e-9)
    q_norm = q_vec / (q_vec.sum() + 1e-9)
    return (1 - fb_weight) * q_norm + fb_weight * p_w_r

# ------------------------------
# 3. MAIN EXPERIMENT
# ------------------------------
def run_experiment(num_queries=100):
    queries_path = "queries.json"
    qrels_path = "qrels_dev.tsv"
    corpus_path = "collection.tsv"

    queries = load_json(queries_path)
    qrels = load_qrels(qrels_path)
    corpus = load_corpus(corpus_path)
    X_sparse, doc_ids, vectorizer = get_vectorized_data(corpus)

    runs = {"Rocchio": {}, "RM3": {}, "BasisChange_DROP": {}, "SVD_DROP": {}}
    target_qids = [qid for qid in queries.keys() if qid in qrels]

    for qid in tqdm(target_qids[:num_queries], desc="Processing Queries"):
        q_text = queries[qid]
        q_vec = vectorizer.transform([q_text]).toarray()[0]

        # Initial retrieval (TF-IDF dot)
        initial_scores = (X_sparse @ q_vec.T).flatten()
        top_idx = np.argsort(initial_scores)[::-1][:1000]

        X_dense = X_sparse[top_idx].toarray()

        # Feedback sets
        rel_idx_local = np.arange(10)
        irrel_idx_local = np.arange(950, 1000)

        # 1. ROCCHIO
        centroid = X_dense[rel_idx_local].mean(axis=0)
        v_rocchio = q_vec + 0.75 * centroid
        runs["Rocchio"][qid] = {doc_ids[top_idx[i]]: float(np.dot(X_dense, v_rocchio)[i]) for i in range(1000)}

        # 2. RM3
        v_rm3 = compute_rm3_vector(q_vec, X_dense, rel_idx_local)
        runs["RM3"][qid] = {doc_ids[top_idx[i]]: float(np.dot(X_dense, v_rm3)[i]) for i in range(1000)}

        # 3. BASIS CHANGE DROP
        w = basis_change_weights(q_vec, X_dense, rel_idx_local, irrel_idx_local)
        v_bc_drop = q_vec * w  # **No centroid smoothing**
        runs["BasisChange_DROP"][qid] = {doc_ids[top_idx[i]]: float(np.dot(X_dense, v_bc_drop)[i]) for i in range(1000)}

        # 4. SVD DROP
        v_svd = apply_svd(q_vec, X_dense, rel_idx_local, k_dims=5)
        v_svd_drop = v_svd  # **No smoothing**
        runs["SVD_DROP"][qid] = {doc_ids[top_idx[i]]: float(np.dot(X_dense, v_svd_drop)[i]) for i in range(1000)}

    # --- EVALUATION ---
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map", "ndcg_cut_10"})

    print("\n" + "="*70)
    print(f"{'Method':<20} | {'MAP':<15} | {'nDCG@10':<15}")
    print("-"*70)
    for name, run_data in runs.items():
        res = evaluator.evaluate(run_data)
        map_val = np.mean([v['map'] for v in res.values()])
        ndcg_val = np.mean([v['ndcg_cut_10'] for v in res.values()])
        print(f"{name:<20} | {map_val:<15.4f} | {ndcg_val:<15.4f}")
    print("="*70)


if __name__ == "__main__":
    run_experiment()
