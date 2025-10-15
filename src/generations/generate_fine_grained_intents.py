import csv
import json
import jsonlines
import random
import pandas as pd
from openai import OpenAI
from datetime import datetime

client = OpenAI(api_key="KEY")

def lmcall(model, temperature, user_prompt):
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=80,
        presence_penalty=0,
        frequency_penalty=0,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.choices[0].message.content.strip()

ks = [50]
examples = []
results = []

with open("training_examples_k=50.csv", mode="r", encoding="ISO-8859-1") as file:
    csvFile = csv.reader(file)
    for line_as_list in csvFile:
        examples.append(line_as_list)

df_test = pd.read_excel("PST.xlsx")

for j, row in df_test.iterrows():
    print(str(j), flush=True)
    #if j==10:
    #	break
    temp = row.to_dict()

    for k in ks:
        predictions = []
        temp["resp"] = ""

        for run in range(3):
            examples_of_run = sorted(random.sample(examples, k), key=lambda x: x[3]) #sort by label, shuffle within label

            user_prompt = """
			A citation context in a scientific paper refers to the specific part of a paper p' where another paper p is mentioned.
			A citation is "impact-revealing" only if the cited work directly contributed to, influenced, or was foundational for the new research for example, if it: 
			provided a method, model, or framework the current study uses or extends, solved a problem that the current study builds on, inspired or shaped the current work's direction.
			If the citation merely compares, lists, defines terms, summarizes prior models, or reviews related work, label it as "other" even if it sounds positive or important.
			Return the output strictly in JSON format: {"description": "<few words>", "label": "impact-revealing" or "other"}
			Below are examples:
			"""

            for ex in examples_of_run:
                user_prompt += (
                    f"<{ex[1]}> => "
                    f'{{"description": "{ex[2]}", "label": "{ex[3]}"}}\n'
                )

            # adding gold examples from each class
            user_prompt += """
			<Eisenberg (2008) integrates digital tools into information literacy and inspires new research directions.> => {"description": "shows foundational influence of Eisenberg's model", "label": "impact-revealing"}
			<Eisenberg (2008) discusses the use of digital tools for education.> => {"description": "general background mention", "label": "other"}
			"""

            context = temp["SS_contexts"].replace("###", " ").strip()
            context = " ".join(context.split())
            user_prompt += f"<{context}> => "

            try:
                resp = lmcall("gpt-4o-mini", 0.0, user_prompt)
                temp["resp"] += "\n" + resp

                #parse response
                label = None
                try:
                    data = json.loads(resp)
                    label = data.get("label", "").lower()
                except Exception:
                    resp_lower = resp.lower()
                    if '"impact-revealing"' in resp_lower or "'impact-revealing'" in resp_lower:
                        label = "impact-revealing"
                    elif '"other"' in resp_lower or "'other'" in resp_lower:
                        label = "other"

                predictions.append(label)

            except Exception as e:
                print(f"Run {run} failed: {e}")

        if len(predictions) == 3:
            temp["our_intent"] = (
                "impact-revealing"
                if predictions.count("impact-revealing") > predictions.count("other")
                else "other"
            )
        else:
            temp["our_intent"] = "error"

        results.append(temp)


f = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
with jsonlines.open(f, "w") as writer:
    writer.write_all(results)

pd.read_json(f, lines=True).to_excel(f.replace(".jsonl", "") + ".xlsx", engine="xlsxwriter")
