import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from FlagEmbedding import FlagAutoModel

model = FlagAutoModel.from_finetuned('BAAI/bge-base-zh-v1.5',
                                      query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                                      use_fp16=True)
# 初始化嵌入模型

def calculate_embeddings(texts, batch_size=8):
    """Generates embeddings for texts in batches due to memory limits."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch)
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)

def calculate_accuracy(pos_indices, top_k_indices):
    """Calculates accuracy for given indices."""
    hits = sum(1 for idx in top_k_indices if idx in pos_indices)
    return hits / len(top_k_indices) if len(top_k_indices) > 0 else 0

def calculate_recall(pos_indices, top_k_indices, pos_num):
    """Calculates accuracy for given indices."""
    hits = sum(1 for idx in top_k_indices if idx in pos_indices)
    return hits / pos_num 

def process_jsonl(file_path, batch_size=8):
    average_accuracies = {'top_5': [], 'top_10': [], 'top_20': []}
    average_recall = {'top_5': [], 'top_10': [], 'top_20': []}

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            query = data["query"]
            pos_samples = data["pos"]
            neg_samples = data["neg"]
            pos_num = len(pos_samples)
            # 生成嵌入
            query_embedding = calculate_embeddings([query], batch_size=batch_size)[0]
            pos_embeddings = calculate_embeddings(pos_samples, batch_size=batch_size)
            neg_embeddings = calculate_embeddings(neg_samples, batch_size=batch_size)

            # 合并pos和neg并计算余弦相似度
            all_embeddings = np.vstack([pos_embeddings, neg_embeddings])
            similarities = cosine_similarity([query_embedding], all_embeddings).flatten()

            # 排序索引，注意这里要从大到小排序
            sorted_indices = np.argsort(similarities)[::-1]

            # 定义pos样本的实际索引范围
            pos_indices_range = set(range(len(pos_samples)))

            # 计算准确率
            average_accuracies['top_5'].append(calculate_accuracy(
                pos_indices_range, sorted_indices[:5]))
            average_recall['top_5'].append(calculate_recall(
                pos_indices_range, sorted_indices[:5],pos_num))
            average_accuracies['top_10'].append(calculate_accuracy(
                pos_indices_range, sorted_indices[:10]))
            average_recall['top_10'].append(calculate_recall(
                pos_indices_range, sorted_indices[:10],pos_num))
            average_accuracies['top_20'].append(calculate_accuracy(
                pos_indices_range, sorted_indices[:20]))
            average_recall['top_20'].append(calculate_recall(
                pos_indices_range, sorted_indices[:20],pos_num))

    # 计算平均准确率
    for key in average_accuracies:
        average_accuracies[key] = np.mean(average_accuracies[key])
        print(f"{key} average accuracy: {average_accuracies[key]}")
    for key in average_recall:
        average_recall[key] = np.mean(average_recall[key])
        print(f"{key} average recall: {average_recall[key]}")

# Example usage
if __name__ == "__main__":
    file_path = 'train_data/bge_test_sample.jsonl'
    process_jsonl(file_path)
