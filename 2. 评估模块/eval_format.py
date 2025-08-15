def extract_json_content(content):
    # 找到json字符串的位置，返回解析后的json对象
    start = content.find('{')
    end = content.rfind('}')
    if start != -1 and end != -1:
        json_str = content[start:end+1]
        try:
            json_ss = json.loads(json_str)
            return json_ss
        except:
            return None
    return None
    
def main(arg1: str) -> dict:
    ct = extract_json_content(arg1)
    if ct is None:
      return '有错误'
    if not ('summary' in ct.keys() and 'algorithm' in ct.keys() and 'compare_result' in ct.keys() and 'keyword_problem' in ct.keys() and 'keyword_algorithm' in ct.keys()):
      return '有错误'
    return '无错误'