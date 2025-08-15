from elasticsearch import Elasticsearch

#es = Elasticsearch("http://localhost:9200")
es = Elasticsearch(
    hosts=["http://localhost:9200"],  # ES服务器地址
    basic_auth=("xxx", "xxx")  # 用户名和密码
)
index_name = "papers-index"
index_body = {
    "settings": {
        "analysis": {
            "analyzer": {
                "default_analyzer": {
                    "type": "custom",
                    "tokenizer": "ik_max_word"
                }
            }
        },
        "number_of_shards": 10,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
          "custom_id": {"type": "text","index": False},
            "summary": {"type": "text", "analyzer": "default_analyzer"},
            "algorithm": {"type": "text", "analyzer": "default_analyzer"},
            "compare_result": {"type": "text", "analyzer": "default_analyzer"},
            "keyword_problem": {"type": "text", "analyzer": "default_analyzer"},
            "keyword_algorithm": {"type": "text", "analyzer": "default_analyzer"},
            "en_org": {"type": "text", "analyzer": "default_analyzer"},
            "ch_org": {"type": "text", "analyzer": "default_analyzer"},
            "title": {"type": "text", "analyzer": "default_analyzer"},
            "created_date": {"type": "date", "format": "yyyy-MM-dd"},
            "embedding": {"type": "dense_vector","dims": 768}
        }
    }
}


es.indices.create(index=index_name, body=index_body)