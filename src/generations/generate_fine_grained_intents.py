!pip install xlsxwriter
!pip install jsonlines
import csv
import json
import jsonlines
import random

ks = [50]
runs = 3
examples = []

with open('/data/training_examples_k=50.csv', mode ='r', encoding="ISO-8859-1")as file:
  csvFile = csv.reader(file)
  for line_as_list in csvFile:
    examples.append(line_as_list)

path = 'path/to/input/'
path2 = 'path/to/output/'

results = []
test_data = []
file = open(path+str(f), 'r')
print(str(f))
citations = file.readlines()

def lmcall(model, temporature, prompt):
  '''
  call to llm
  '''

def main():
  for cit in citations:
    json_object = json.loads(cit)
    test_instance = {}
    if "context" not in json_object:
      continue
    if len(json_object["context"]) == 0:
      continue

    test_instance["cited_paper_id"] = json_object["cited_id"]
    test_instance["cited_paper_title"] = json_object["cited_title"]
    test_instance["cited_paper_year"] = json_object["cited_year"]

    test_instance["citing_paper_id"] = json_object["citing_id"]
    test_instance["citing_paper_title"] = json_object["citing_title"]
    test_instance["citing_paper_year"] = json_object["citing_year"]
    test_instance["context"] = ""
    for c in json_object["context"]:
      test_instance["context"] = test_instance["context"] + c
    test_data.append(test_instance)

  for test_instance in test_data: 
    for k in ks:
      temp = test_instance.copy()
      for run in range(0, runs):
        random.shuffle(examples)
        examples_of_run = examples[:k]
        prompt ="""
        A citation context in a scientific paper refers to the specific part of a paper p′ where
        another paper p is mentioned, including the surrounding sentences or paragraphs that
        explain how and why p is relevant to p′.
        A citation context with an impact-revealing intent is a type of citation in scientific
        writing that highlights the significance or influence of a previously published work, often
        emphasizing its contribution or importance to the current research or the broader field,
        e.g., its role in inspiring, motivating, supporting, filling gaps, critically analyzing, or
        contributing methods, tools, data, extensions, or benchmarks for the current research.
        Other types of intents include a reference to prior work in a scientific paper that provides
        background or context without emphasizing the impact, significance, or influence of
        the cited work. It acknowledges the source in a routine or supporting role rather than
        showcasing its importance to the research.
        Given a citation context, describe, in a few words, the intention behind this citation phrase.
        Then, decide on the category of this intention. In particular, whether the intention behind
        this citation phrase is impact-revealing or not (i.e., incidental or that there isn’t enough
        information to realize the real intention behind it). For the intention category, only return
        one of the following two labels impact-revealing or other .
        Below are examples:
        """
        for ex in examples_of_run:
          prompt+="<"+ex[1]+"> => <\""+ex[2]+"\"|\""+ex[3]+"\">\n"

        temp["prompt_run"+str(run)] = "<"+temp["context"]+"> =>"

        try:
          response=lmcall("gpt4omini", 0, prompt+"<"+temp["context"]+"> =>")
          resp = str(response.content)
          temp["response"+str(run)] = resp
          #print(resp)
          resp = resp.replace("\"", "").replace(">","").replace("<","")
          if len(resp.split("|"))>=2:
            temp["intentclass"+str(run)] = resp.split("|")[1].strip()
            temp["intentdescr"+str(run)] = resp.split("|")[0].strip()
          elif len(resp.split("=>"))>=2:
            temp["intentclass"+str(run)] = resp.split("=>")[1].strip()
            temp["intentdescr"+str(run)] = resp.split("=>")[0].strip()
          else:
            temp["intentclass"+str(run)] = "ERROR"
            temp["intentdescr"+str(run)] = "ERROR"

          results.append(temp)
        except:
          temp["response"+str(run)]= ""
          temp["intentclass"+str(run)] = ""
          temp["intentdescr"+str(run)] = ""
          results.append(temp)

  with jsonlines.open(path2+str(f), 'w') as writer:
    writer.write_all(results)
  import pandas
  pandas.read_json(path2+str(f), lines=True).to_excel(path2+str(f).replace('.jsonl','')+".xlsx", engine='xlsxwriter')

if __name__ == '__main__':
  main()
