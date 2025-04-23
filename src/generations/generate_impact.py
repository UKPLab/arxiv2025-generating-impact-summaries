!pip install jsonlines
import jsonlines
!pip install xlsxwriter
import json
import pandas as pd
import math

def lmcall_schema(model, temperature, prompt):
  json_schema =  
      {"name": "impact_statement",
        "schema": {"type": "object",
                "properties": {
                "input_paper_info": {
                    "type": "object",
                    "items": {
                    "input_paper_id": "id of input paper",
                    "input_paper_title": "title of input paper",
                    "input_paper_year": "year of input paper"}},
                "impact_periods":{
                  "type": "array",
                  "items": {
                      "impact_period": "start year - end year",
                      "aspect_of_period": "the dominating citation intent(s) of that period",
                      "impact_description": "a paragraph to describe the impact of that period",
                      "evidence": "citing papers from that period to back up the described impact aspect"}}
                }
            }
    }

  '''
  call_llm(model, temperature, prompt, json_schema)
  '''

def main():
  df = pd.read_excel('/path/to/output/file/from/generate_fine_grained_intents')
  exc = open("exceptions.tsv", "w")

  results = []
  in_file = open('/path/to/paperids/and/metadata', 'r')

  for line in in_file:
    parts = line.split("\t")
    new_row = {}
    new_row["id"] = parts[0] #paper id
    new_row["title"] = parts[1] #paper title
    new_row["year"] = str(int(parts[2])) #paper year
    new_row["citations"] = ""
    new_row["citations_title"] = {}
    local_df = df.loc[(df['cited_paper_id'] == new_row['id']) & (df['intentclass0'] == 'impact-revealing')]
    local_df = local_df.sample(frac=1).reset_index(drop=True)
    if local_df.shape[0] == 0:
      continue
    #print(local_df)
    citation_id = 0
    new_row["shorter_ids"] = []
    for index, row in local_df.iterrows():
      if math.isnan(float(row['citing_paper_year'])):
        continue
      new_row["citations"] = new_row["citations"] + "<" + "citation_id:"+str(citation_id) + ", Citation_title: \"" + str(row['citing_paper_title']) + "\", " + ", Citation_context: \"" + str(row['context']) + "\", Year: " + str(int(row['citing_paper_year']))  + ", Citation_intent: \"" + str(row['intentdescr0']) +"\">\n"
      new_row["shorter_ids"].append(str(row['citing_paper_id']))
      new_row["citations_title"][citation_id] = "["+str(citation_id)+"]: <a href=\"https://www.semanticscholar.org/paper/"+str(row['citing_paper_id'])+"\" target=\"_blank\">"+row['citing_paper_title']+"</a>"
      citation_id = citation_id + 1

    prompt = '''
    The scientific impact summary of a research paper describes the impact a given
    paper had on other papers, including both praise and critique. To understand the impact
    of a paper, one needs to understand how exactly it has been utilized and discussed
    by other papers. This is normally referred to as citation intents. One also needs to
    understand the evolution of the impact and citation intents over time.
    Given an input paper’s title, its publication year, and its citation context, describe the
    impact of that paper.
    The citation context includes five components: <citation ID, citation title, citation context, citation year, citation intent>.

    Given the input paper with id {paper_id} titled {title} published in {year}, and the following list of papers citing it:
    <citation ID, citation title, citation phrase, citation year, citation intent description>\n
    {citation_context}

    Generate an impact statement about the input paper.
    '''.format(citation_context = new_row["citations"], title=new_row["title"], year=new_row["year"], paper_id= new_row["id"])

    new_row["prompt"] = prompt
    try:
      res = lmcall_schema("gpt-4o", 0, prompt)
    except:
        print("error with paper "+new_row["id"])
        continue
    new_row["response"] = res.content
    results.append(new_row)


  f = 'path/to/output/'

  with jsonlines.open(f+".jsonl", 'w') as writer:
    writer.write_all(results)

  import pandas
  pandas.read_json(f+".jsonl", lines=True).to_excel(str(f).replace('.jsonl','')+".xlsx", engine='xlsxwriter')

if __name__ == '__main__':
  main()
