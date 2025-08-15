def get_emb_query(tt_modify, org, embedding):
    query_st = {"query": {"bool": {"must": []}},"size": size,"_source": ["custom_id","title", "summary", "en_org","update_date","authors","algorithm","compare_result"]}
    if tt_modify != "":
        query_time={"range": {"create_time": {"gte": tt_modify}}}
        query_st["query"]["bool"]["must"].append(query_time)
    if org != '无':
        query_org={"multi_match": {"query": org,"fields": ["en_org", "ch_org"]}}
        query_st["query"]["bool"]["must"].append(query_org)
    query_text = {"knn": {
          "field": "emb",
          "query_vector": [],
          "k": 50,
          "num_candidates": 100
        }
      }
    query_st["query"]["bool"]["must"].append(query_text)
    return query_st