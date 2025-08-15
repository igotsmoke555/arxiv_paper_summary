def get_search_query(tt_modify, org, query_key):
    query_st = {"query": {"bool": {"must": []}},"size": size,"_source": ["custom_id","title", "summary", "en_org","update_date","authors","algorithm","compare_result"]}
    if tt_modify != "":
        query_time={"range": {"create_time": {"gte": tt_modify}}}
        query_st["query"]["bool"]["must"].append(query_time)
    if org != '无':
        query_org={"multi_match": {"query": org,"fields": ["en_org", "ch_org"]}}
        query_st["query"]["bool"]["must"].append(query_org)
    query_text = {"knn": {
            "query": query_key,
            "fields": [
              "keyword_problem^2",
              "keyword_ct^2",
              "summary",
              "algorithm"
            ]
          }}
    query_st["query"]["bool"]["must"].append(query_text)
    return query_st