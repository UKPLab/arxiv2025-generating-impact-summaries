import os
import requests
import pandas as pd
from ast import literal_eval
import ast
!pip install jsonlines
import json
import jsonlines
import pandas as pd

df = pd.read_excel('themes.xlsx')

def get_full_statement(decomposed_statement, paper_name):
    statement = ""
    for impact_desc in decomposed_statement:
        period_key = 'impact_period' if 'impact_period' in impact_desc else 'impact_periods'
        if period_key in impact_desc and 'aspect_of_period' in impact_desc:
            statement += f"{impact_desc['aspect_of_period']}" + " "
        if 'impact_description' in impact_desc:
            statement += f"{impact_desc['impact_description']}" + " "

    return statement



variant_dirs = os.scandir('/variants_results/json_files')
variant_dirs = [d.path for d in variant_dirs if d.is_dir() and '=' in d.name]

results = []

for variant_dir in variant_dirs:
  print(variant_dir)
  variant_dir_statements_path = os.path.join(variant_dir, 'statements_with_corpus.csv')
  eval_data = pd.read_csv(variant_dir_statements_path, dtype={'decomposed_statements': object})
  eval_data['decomposed_statements'] = eval_data['decomposed_statements'].apply(ast.literal_eval)
  variant_name = os.path.basename(variant_dir)
  run_name = 'run1'
  if 'run1' in variant_name:
    variant_name = variant_name.replace('run1_', '')
  if 'run2' in variant_name:
    variant_name = variant_name.replace('run2_', '')
    run_name = 'run2'
  for _, statement in eval_data.iterrows():
    print(_)
    result = {}
    result['variant'] = variant_name
    result['run'] = run_name
    result['title'] = statement['title']
    result['id'] = statement['id']
    decomposed_statement = statement['decomposed_statements']['impact']
    paper_name = statement['title']
    local_df = df.loc[(df['id'] == statement['id'])]
    full_themes = []
    for index, row in local_df.iterrows():
      full_themes = literal_eval(row['themes'].replace('```python','').replace('```','').replace('\n',''))
    statement_text = get_full_statement(decomposed_statement, paper_name)
    result['full_themes'] = full_themes
    result['statement'] = statement_text

    prompt= "Given this list of themes in the format of a python list: "+full_themes+" determine how many of these themes were implicitly or explicitly mentioned in this summary " + statement_text + ", and list them".
    result['prompt'] = prompt
    try:
      response=lmcall_coverage("gpt4omini", 0, prompt)
      resp = str(response.content).strip().lower()
      result['response'] = resp
      resp_json = json.loads(resp)
      result['coverage'] = resp_json['result']['number_of_themes_covered']/len(full_themes)
    except:
      result['response'] = response

    results.append(result)

with jsonlines.open('coverage.jsonl', 'w') as writer:
  writer.write_all(results)
import pandas
pandas.read_json('coverage.jsonl', lines=True).to_excel("coverage.xlsx")
