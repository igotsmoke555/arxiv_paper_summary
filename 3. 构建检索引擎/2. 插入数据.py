import json
import time
from elasticsearch import Elasticsearch

#es = Elasticsearch("http://localhost:9200")
es = Elasticsearch(
    hosts=["http://localhost:9200"],  # ES服务器地址
    basic_auth=("xxx", "xxx")  # 用户名和密码
)
def bulk_import(es, file_path, index_name):
    num = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            num += 1
            data = json.loads(line)
            es.index(
                index=index_name,
                #id="my_document_id",
                document=data,
            )
            print(num)
            if num%1000 == 0:
                time.sleep(0.2)

bulk_import(es, "/home/ecs-user/data/merged_summary_org_infos_emb_data.jsonl", 'papers-index')