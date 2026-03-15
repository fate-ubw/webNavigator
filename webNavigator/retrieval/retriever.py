import os
import time
import pickle
from functools import wraps
from typing import Dict, Any, List, Tuple, Callable
import argparse
import requests
import numpy as np
from dotenv import load_dotenv
import pprint

load_dotenv()
JINA_API_URL = "https://api.jina.ai/v1/embeddings"
JINA_MODEL = "jina-embeddings-v4"
QUERY_TASK = "retrieval.query"
TIME_OUT = 100
MAX_RETRIES = 10


def timing(func: Callable) -> Callable:
    """Timing decorator: measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[{func.__name__}] elapsed: {elapsed:.3f}s")
        return result
    return wrapper


class JinaRetriever:
  
    def __init__(self, args):
        self.api_key = getattr(args,"api_key", os.getenv("JINA_API_KEY"))
        if not self.api_key: 
            # main can get None, but here can get the env key
            self.api_key = os.getenv("JINA_API_KEY")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        self.model = getattr(args.retriever, "model", JINA_MODEL)
        # load existing embeddings
        self.return_multivector = getattr(args.retriever, "return_multivector", True)
        self.names, self.embs = self._load_existing_embeddings(getattr(args, "persistence_file"), self.return_multivector)
        # retrieval setting
        self.topk = getattr(args.retriever, "topk", 50)

    def _load_existing_embeddings(self, path: str, return_multivector: bool = False) -> Tuple[List[str], List[List[float]]]:
        if not path or not os.path.exists(path):
            return [], []
        with open(path,"rb") as f:
            data = pickle.load(f)
        # Split names and embeddings for simpler downstream NumPy processing.
        names, vecs = [], []
        print(f"Loading {len(data)} embeddings from {path}")
        for item in data:
            if return_multivector:
                if isinstance(item,dict) and "image_name" in item and "multi_embeddings" in item:
                    names.append(item["image_name"])
                    vecs.append(item["multi_embeddings"])
                else:
                    raise ValueError("Invalid embedding data")
            else:
                if isinstance(item,dict) and "image_name" in item and "embedding" in item:
                    names.append(item["image_name"])
                    vecs.append(item["embedding"])
                else:
                    raise ValueError("Invalid embedding data")
        # --> end of for loop
        return names, vecs

    @timing
    def encode_text(self, query:str, task:str = QUERY_TASK, return_multivector:bool = False) -> List[float]:
        "single text input, next version will support batch encoding"
        if not query:
            raise ValueError("query is required")
        payload = {
            "model": self.model,
            "task": task or QUERY_TASK,
            "input": [{"text": query}],
        }
        if return_multivector:
            payload["return_multivector"] = True
        
        last_error = None
        for attemp in range(1, MAX_RETRIES + 1):
            try:
                response = requests.post(
                        JINA_API_URL, 
                        headers=self.headers, 
                        json=payload, 
                        timeout=TIME_OUT)

                if response.ok:
                    output = response.json()
                    data = output.get("data", [])
                    query_len = output.get('usage')['total_tokens']
                    if not data:
                        raise ValueError("No data returned from Jina API")
                    if return_multivector:
                        return data[0]["embeddings"], query_len
                    else:
                        return data[0]["embedding"], query_len
            except requests.RequestException as e:
                last_error = e
                print(f"Failed to encode text: {e}")
                time.sleep(1 if attemp == 1 else 2)
        raise RuntimeError(f"Failed to encode text: {last_error}")
    
    @timing
    def search(self,query:str) -> list[dict]:
        # search the database for the most similar images to the query
        q_embed, len = self.encode_text(query)
        # Compute similarity against all indexed entries.
        results = []
        for name, emb in zip(self.names, self.embs):
            score = cosine_similarity(q_embed, emb)
            results.append((name, score))
        # Sort by similarity score.
        results.sort(key=lambda x: x[1], reverse=True)
        topk_results = results[:self.topk]
        return [{"image_name":name, "score":score} for name, score in topk_results]
    
    @timing
    def rerank(self, query:str ) -> list[dict]:
        # rerank the results
        q_embed, len = self.encode_text(query, return_multivector=True)
        results = []
        for name, emb in zip(self.names, self.embs):
            score = late_interaction_score(q_embed, emb)
            results.append((name, score))
        # Sort by similarity score.
        results.sort(key=lambda x: x[1], reverse=True)
        topk_results = results[:self.topk]
        return [{"image_name":name, "score":score} for name, score in topk_results]

    def __repr__(self) -> str:
        return f"JinaRetriever(model={self.model}, return_multivector={self.return_multivector}, topk={self.topk})"

def cosine_similarity(a:List[float], b:List[float]) -> float:
    vec_a = np.asarray(a, dtype=np.float32)
    vec_b = np.asarray(b, dtype=np.float32)
    norm_product = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if norm_product == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / norm_product)


def late_interaction_score(
    query_multi_vector: List[List[float]],
    doc_multi_vector: List[List[float]]
) -> float:
    """
    Calculates the late interaction (ColBERT-style) score between a query and a document.

    This function computes the sum of max similarities (MaxSim). For each vector in the query's
    multi-vector, it finds the maximum cosine similarity with any vector in the document's
    multi-vector. The final score is the sum of these maximum similarities.

    Args:
        query_multi_vector: A list of embedding vectors for the query.
        doc_multi_vector: A list of embedding vectors for the document.

    Returns:
        The final late interaction score as a float.
    """
    # 1. Handle empty inputs to avoid errors.
    if not query_multi_vector or not doc_multi_vector:
        return 0.0

    # 2. Convert lists of vectors to NumPy arrays for efficient computation.
    # We expect shapes: (num_query_tokens, dim) and (num_doc_tokens, dim)
    query_vecs = np.array(query_multi_vector, dtype=np.float32)
    doc_vecs = np.array(doc_multi_vector, dtype=np.float32)

    # 3. Normalize the vectors to unit length. This is a crucial optimization.
    # When vectors are normalized, their dot product is equivalent to their cosine similarity.
    # This avoids repeated norm calculations inside the loop.
    query_vecs_normalized = query_vecs / np.linalg.norm(query_vecs, axis=1, keepdims=True)
    doc_vecs_normalized = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)

    # 4. Compute the cosine similarity matrix.
    # This is the most computationally intensive step, but it's a single, highly optimized
    # matrix multiplication operation. The resulting matrix will have a shape of
    # (num_query_tokens, num_doc_tokens).
    similarity_matrix = np.dot(query_vecs_normalized, doc_vecs_normalized.T)

    # 5. Find the maximum similarity for each query vector.
    # We take the maximum value along axis 1 (across the document tokens for each query token).
    # The result is a 1D array of shape (num_query_tokens,).
    max_similarities = np.max(similarity_matrix, axis=1)

    # 6. Sum these maximum similarities to get the final score.
    final_score = np.sum(max_similarities) / len(max_similarities)

    return float(final_score)


if __name__ == "__main__":
    args = argparse.Namespace(
        persistence_file="./webNodes/run_id-20251026-151146-admin/image_embedding-multivector.pkl",
    )
    # recall - no interaction
    # retriever = JinaRetriever(args)
    # results = retriever.search("What is the top-1 best-selling product in 2022") # 
    # pprint.pprint(results)
    # rerank - late interaction
    args.return_multivector = True
    retriever = JinaRetriever(args)
    query = "Admin dashboard sales reports or analytics page with a focus on yearly performance or best-selling products." # ver 4.0
    query = "What is the top-1 best-selling product in 2022" # original query
    query = "Page showing sales analytics or reports highlighting the best-selling products, potentially with a filter for the year 2022" # ver 5.0
    query = "The immediate sub-goal is to locate the sales or analytics section that contains the data for the best-selling products in 2022. The current admin dashboard page does not provide any specific data or links about sales. I am looking for a page with charts, tables, or a list that highlights product sales performance, with filters or headers for years, such as 2022."
    print(f"Query: {query}")
    results = retriever.rereank(query)
    pprint.pprint(results)