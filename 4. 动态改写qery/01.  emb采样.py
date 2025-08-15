import json
import numpy as np
import random
from sklearn.cluster import KMeans
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform

from FlagEmbedding import FlagAutoModel
model = FlagAutoModel.from_finetuned('BAAI/bge-large-zh-v1.5',
                                     query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                                     use_fp16=True,
                                     devices=['cuda:0']) # Setting use_fp16 to True speeds up computation with a slight performance degradation

def compute():
    # Read queries from JSONL file
    input_file = 'rewrite_query_output_pair.jsonl'
    queries = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            query_data = json.loads(line)
            queries.append(query_data)
    
    # Extract ori_query for embedding
    ori_queries = [q['ori_query'] for q in queries]
    
    # Batch size for embedding
    batch_size = 16
    
    # Generate embeddings in batches
    embeddings = []
    for start_idx in tqdm(range(0, len(ori_queries), batch_size), desc="Generating embeddings"):
        batch_queries = ori_queries[start_idx:start_idx + batch_size]
        batch_embeddings = model.encode(batch_queries)
        embeddings.extend(batch_embeddings)
    
    embeddings = np.array(embeddings)
    
    # Calculate cosine similarity matrix (just for verification)
    similarity_matrix = cosine_similarity(embeddings)
    
    # Define number of clusters
    num_clusters = 10000  # 根据数据特点进行调整
    
    # Perform KMeans clustering
    # KMeans does not directly use cosine similarity, so we don't set cosine as metric here
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # 下采样Downsample: randomly keep one query from each cluster
    downsampled_queries = []
    unique_labels = set(cluster_labels)
    for cluster in unique_labels:
        # Get queries belonging to the current cluster
        cluster_indices = np.where(cluster_labels == cluster)[0]
        cluster_queries = [queries[idx] for idx in cluster_indices]
    
        # Randomly select one query pair from the cluster
        sampled_query = random.choice(cluster_queries)
    
        # Collect sampled query
        downsampled_queries.append(sampled_query)
    
    # Save downsampled queries to JSONL
    output_file = 'downsampled_queries.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for query in downsampled_queries:
            f.write(json.dumps(query, ensure_ascii=False) + '\n')
    
    print(f"Downsampled queries saved to {output_file}")
if __name__ == '__main__':
    compute()