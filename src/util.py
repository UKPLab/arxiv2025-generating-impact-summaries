import json
import logging
import os
import re
from datetime import datetime
from typing import Dict, Tuple, List
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd

unique_field_of_study = ['Computer Science', 'Medicine', 'Psychology']


def request_openai_batch_completions(prompts: Dict[str, str], max_tokens: int, temperature: float, batch_idx: int,
                                     output_path: str, client, engine: str) -> str:
    batch_requests = []
    for prompt_id, prompt in prompts.items():
        batch_entry = {"custom_id": prompt_id, "method": "POST",
                       "url": "/v1/chat/completions",
                       "body": {"model": engine, "max_tokens": max_tokens, "temperature": temperature,
                                "messages": [{"role": "user", "content": prompt}]}}
        batch_requests.append(batch_entry)

    batch_requests = pd.DataFrame(batch_requests)
    batch_requests_file = os.path.join(output_path, f'batch_{batch_idx}_requests.json')
    batch_requests.to_json(batch_requests_file, lines=True, orient='records')

    batch_input_file = client.files.create(
        file=open(batch_requests_file, "rb"),
        purpose="batch"
    )

    batch_out = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": f'Batch {batch_idx}',
        }
    )

    return batch_out.id


def get_openai_batch_completions(batch_id: str, client) -> Tuple[Dict[str, str], str]:
    batch_status = client.batches.retrieve(batch_id)
    query_responses_by_id = {}
    if batch_status.status == "completed":
        batch_response = client.files.content(batch_status.output_file_id).text
        query_responses = [json.loads(r) for r in batch_response.strip().split('\n')]
        query_responses_by_id = {}
        for response in query_responses:
            response_content = response['response']['body']['choices'][0]['message']['content']
            query_responses_by_id[response['custom_id']] = response_content
    elif batch_status.status == "failed":
        raise Exception(f'Batch {batch_id} failed')

    return query_responses_by_id, batch_status.status


def create_out_dir(output_dir: str) -> None:
    """Create an output directory if it does not exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def setup_default_logger(output_dir: str) -> logging.Logger:
    """Set up a logger that writes to output_dir"""
    logs_dir = os.path.join(output_dir, 'logs')
    create_out_dir(logs_dir)
    logger = logging.getLogger(__name__)
    log_format = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = os.path.join(logs_dir, f'{timestamp}.log')
    logger.info(f'Writing logs to {log_file}')
    file_handler = logging.FileHandler(log_file)
    logger.addHandler(file_handler)
    return logger


def encode_sources(sentence_encoder, corpus):
    flattened_sources = []
    for _, row in corpus.iterrows():
        for context in row['contexts']:
            flattened_sources.append(
                {'id': row['id'], 'year': row['year'], 'title': row['title'], 'context': context.replace('\n', ' ')})

    texts = [source['context'] for source in flattened_sources]
    encodings = sentence_encoder.encode(texts, show_progress_bar=True)
    sources = [{'id': source['id'], 'year': source['year'], 'title': source['title'], 'context': source['context'],
                'encoding': encoding} for source, encoding in zip(flattened_sources, encodings)]

    return sources


def get_claim_start_end_years(claim_period):
    claim_period = claim_period.lower().strip()
    current_year = 2025
    replacements = ['current', 'present', 'now', 'upcoming', 'and onward', 'and beyond', 'onwards', 'future', 'ongoing',
                    'onward']
    for replacement in replacements:
        claim_period = claim_period.replace(replacement, str(current_year))
    pattern = r"(\d{4})(?:\s*-\s*(\d{4}))?"

    match = re.search(pattern, claim_period)
    if match:
        start_year = match.group(1)
        end_year = match.group(2) or start_year
        return int(start_year), int(end_year)
    else:
        return None, None


def log_clusters(sources_by_cluster):
    logstr = '\n---CLUSTERS---\n'
    nr_big_clusters = 0
    for cluster_id, cluster_sources in sources_by_cluster.items():
        if len(cluster_sources) > 1:
            logstr += f'---{cluster_id}\n'
            for source in cluster_sources:
                logstr += f'{source["claim"]}\n'
            nr_big_clusters += 1
    logstr += '---\n'
    if nr_big_clusters > 0:
        print(logstr)


def cluster_sources(sources: List[dict], threshold: float):
    if len(sources) < 2:
        return sources
    embeddings = [source['encoding'] for source in sources]
    clusters = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, linkage='average',
                                       metric="cosine").fit(embeddings)
    sources_by_cluster = {}
    for i, cluster_id in enumerate(clusters.labels_):
        if cluster_id not in sources_by_cluster:
            sources_by_cluster[cluster_id] = []
        sources_by_cluster[cluster_id].append(sources[i])

    # log_clusters(sources_by_cluster)

    clusters_representatives = []
    for cluster_id, cluster_citations in sources_by_cluster.items():
        cluster_embeddings = np.array([citation['encoding'] for citation in cluster_citations])
        centroid = np.mean(cluster_embeddings, axis=0)
        closest_index = np.argmin([np.dot(centroid, embedding) /
                                   (np.linalg.norm(centroid) * np.linalg.norm(embedding))
                                   for embedding in cluster_embeddings])
        clusters_representatives.append(cluster_citations[closest_index])
    return clusters_representatives


def get_sources_textual_representation(sources):
    sources_text = ''
    for source in sources:
        sources_text += f'<{source["title"]}>: "{source["context"]}"\n'
    return sources_text[:-1]


def year_is_in_time_period(year, time_period):
    return time_period[0] <= year <= time_period[1]


def parse_citations(text):
    text = text.replace(', ,', ',')

    if 'Citation_intent' in text:
        citation_pattern = re.compile(
            r'<citation_id:(\d+), Citation_title: "(.*?)", Citation_context: "(.*?)", Year: (\d+), Citation_intent: "(.*?)"',
            re.DOTALL
        )
    else:
        citation_pattern = re.compile(
            r'<citation_id:(\d+), Citation_title: "(.*?)", Citation_context: "(.*?)", Year: (\d+)',
            re.DOTALL
        )

    citations = {}

    for match in citation_pattern.finditer(text):
        citation_id = match.group(1)
        citations[citation_id] = {
            "title": match.group(2).strip(),
            "context": match.group(3).replace("\n", " ").strip(),
            "year": int(match.group(4)),
            "intent": match.group(5) if 'Citation_intent' in text else None
        }

    if not citations:
        print(f'No citations found in text: {text}')

    return citations


def parse_evidence(evidence):
    if isinstance(evidence, str):
        evidence = [evidence]
    return [str(e).replace('>', '').replace('<', '').replace('citation_id', '').split(':')[-1].strip() for e in
            evidence]


def get_field_of_study(paper_info_path):
    field_of_study = {}
    paper_info = pd.read_csv(paper_info_path, sep='\t', names=['id', 'title', 'year', 'citations', 'field_of_study'])
    for _, row in paper_info.iterrows():
        field_of_study[row['id']] = row['field_of_study']
    return field_of_study
