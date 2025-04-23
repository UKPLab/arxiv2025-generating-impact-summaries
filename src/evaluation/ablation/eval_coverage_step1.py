import os
import requests
import pandas as pd
import ast
!pip install jsonlines
import jsonlines
import pandas as pd

#needed to get impact-revealing context
df = pd.read_excel('/path/to/output/file/from/generate_fine_grained_intents')

results = []

in_file = open('/path/to/paperids/and/metadata', 'r')

for line in in_file:
  parts = line.split("\t")
  new_row = {}
  new_row["id"] = parts[0]
  local_df = df.loc[(df['cited_paper_id'] == new_row["id"]) & (df['intentclass0'] == 'impact-revealing')]
  intents = ""
  for index, row in local_df.iterrows():
    intents = intents + "\"" + str(row['intentdescr0']) +"\"\n"
  new_row["intents"] = intents
  prompt= "Given this list of phrases " + intents+", cluster highly similar phrases and give a label (expressive theme) for every cluster."
  try:
    response=lmcall("gpt4omini", 0, prompt)
    new_row["themes"] = response.content.strip()
    print(response.content.strip())
  except:
    new_row['themes'] = resp
  results.append(new_row)

with jsonlines.open('themes.jsonl', 'w') as writer:
  writer.write_all(results)
import pandas
pandas.read_json('themes.jsonl', lines=True).to_excel("themes.xlsx")
