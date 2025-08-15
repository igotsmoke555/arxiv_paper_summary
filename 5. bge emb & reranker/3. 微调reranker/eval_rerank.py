import json
import os
from FlagEmbedding import FlagReranker

def calculate_accuracy(scores, pos_indexes, top_k):
    top_k_indexes = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    correct_hits = sum([1 for idx in top_k_indexes if idx in pos_indexes])
    return correct_hits / top_k

def evaluate_jsonl(jsonl_file_path):
    # Initialize the model
    model = FlagReranker(
        'BAAI/bge-reranker-base',
        use_fp16=True,
        batch_size=128,
        query_max_length=256,
        max_length=256,
        devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        cache_dir=os.getenv('HF_HUB_CACHE', None),
    )

    top5_accuracy = []
    top10_accuracy = []
    top20_accuracy = []

    with open(jsonl_file_path, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            query = data["query"]
            pos_docs = data["pos"]
            neg_docs = data["neg"]

            # Create pairs for positive and negative documents
            pairs = [[query, doc] for doc in pos_docs + neg_docs]
            pos_indexes = list(range(len(pos_docs)))

            # Compute scores for all pairs
            scores = model.compute_score(pairs)

            # Calculate top-k accuracy
            top5_accuracy.append(calculate_accuracy(scores, pos_indexes, top_k=5))
            top10_accuracy.append(calculate_accuracy(scores, pos_indexes, top_k=10))
            top20_accuracy.append(calculate_accuracy(scores, pos_indexes, top_k=20))

    # Calculate mean accuracy for each top-k level
    mean_top5_accuracy = sum(top5_accuracy) / len(top5_accuracy)
    mean_top10_accuracy = sum(top10_accuracy) / len(top10_accuracy)
    mean_top20_accuracy = sum(top20_accuracy) / len(top20_accuracy)

    return mean_top5_accuracy, mean_top10_accuracy, mean_top20_accuracy

# Example usage
if __name__ == '__main__':
    mean_top5, mean_top10, mean_top20 = evaluate_jsonl('bge_test_sample.jsonl')
    print(f"Top 5 accuracy: {mean_top5}")
    print(f"Top 10 accuracy: {mean_top10}")
    print(f"Top 20 accuracy: {mean_top20}")
