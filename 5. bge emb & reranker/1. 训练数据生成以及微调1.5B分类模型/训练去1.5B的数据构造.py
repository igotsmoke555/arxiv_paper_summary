import math
from FlagEmbedding import FlagModel
import json
from elasticsearch import Elasticsearch
import random
import time
# 配置 Elasticsearch 客户端
#es = Elasticsearch("http://47.121.219.144:9200")
es = Elasticsearch("http://localhost:9200")
index_name = "papers-index"
model_dir = '/home/ecs-user/code/es_search/models/bge-base-zh-v1.5'
model = FlagModel(model_dir, 
                  query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                  use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
# 输入文件名
input_file = 'rewrite_query_output.jsonl'

# 输出文件名
output_file = 'query_get_es/rewrite_query_get_es_output'

# 读取 JSONL 文件并处理每一行
samples_all = []
index = 0
error = 0
save_num = 500
ii = 0
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        index += 1
        data = json.loads(line)
        query_ct = data.get('query_ct', '')
        if query_ct == '':
            continue
        embedding = model.encode(query_ct)

        # 构造 Elasticsearch 查询
        es_query = {
            "size": 100,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_emb, 'emb')",
                        "params": {"query_emb": embedding}
                    }
                }
            },
            "_source": ["custom_id", "title", "summary", "en_org", "update_date", "authors", "algorithm", "compare_result"]
        }

        # 查询 Elasticsearch
        try:
            response = es.search(index="papers-index", body=es_query)  # 替换为您的索引名称
            error = 0
        except Exception as e:
            print(e)
            time.sleep(5)
            error += 1
            if error>=20:
                break
            continue
        hits = response.get('hits', {}).get('hits', [])
        print(len(samples_all))
        # 处理查询结果
        if len(hits) == 0:
            continue
        results = []
        for hit in hits:
            source = hit.get('_source', {})
            score = hit.get('_score')
            result = {
                "custom_id": source.get("custom_id"),
                "title": source.get("title"),
                "summary": source.get("summary"),
                "en_org": source.get("en_org"),
                "update_date": source.get("update_date"),
                "authors": source.get("authors"),
                "algorithm": source.get("algorithm"),
                "compare_result": source.get("compare_result"),
                "score": score
            }
            results.append(result)

        # 将原始字段和新字段保存到结果中
        data['query_results'] = results
        samples_all.append(data)
        print(len(samples_all))
        if len(samples_all)%save_num == 0:
            ii += 1
            file_name = output_file+'_'+str(ii)+'.json' 
            with open(file_name, 'w', encoding='utf-8') as f:
                json.dump(samples_all, f, ensure_ascii=False, indent=4)
            samples_all = [] 

# 将结果写入新的 JSON 文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(samples_all, f, ensure_ascii=False, indent=4)

print(f"查询结果已保存到 {output_file}")