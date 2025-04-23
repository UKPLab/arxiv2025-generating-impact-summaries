import os
import logging
import time
from typing import List, Dict
import requests
from tqdm import tqdm


class SemanticScholarAPI:
    """Interact with the Semantic Scholar API to get information about papers."""

    def __init__(self, api_key_path: str, logger: logging.Logger, request_timeout_sec: int):
        self.logger = logger
        self.api_key = None
        self.request_timeout_sec = request_timeout_sec
        self.set_semantic_scholar_api_key(api_key_path)

    def set_semantic_scholar_api_key(self, api_key_path: str) -> None:
        """Set the Semantic Scholar API key from a file."""
        assert os.path.exists(api_key_path), f'API key path {api_key_path} does not exist'
        with open(api_key_path, 'r', encoding="utf-8") as data_file:
            self.api_key = data_file.read().strip()

    def get_paper_references(self, paper_id: str):
        """Get the references of a paper."""
        page_size = 1000
        offset = 0
        backoff = 5

        req_kwargs = {
            'url': f'https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations',
            'headers': {'X-API-KEY': self.api_key},
            'params': {'fields': 'paperId,title,abstract,year,externalIds,contexts',
                       'limit': page_size,
                       'offset': offset},
        }

        while True:
            try:
                rsp = requests.get(**req_kwargs, timeout=self.request_timeout_sec)
                rsp.raise_for_status()
            except requests.exceptions.HTTPError as exception:
                self.logger.warning(f'Error fetching references for paper {paper_id}: {exception}')
                self.logger.info(f'Retrying in {backoff} seconds')
                time.sleep(backoff)
                backoff = min(backoff * 2, 64)
                continue
            page = rsp.json()["data"]
            self.logger.info(f'Got {len(page)} references for paper {paper_id}!')
            for element in page:
                yield element

            if len(page) < page_size:
                break  # no more pages
            req_kwargs['params']['offset'] += page_size
            self.logger.info(
                f'Getting next page (offset={req_kwargs["params"]["offset"]}) of references for paper {paper_id}...')

    def get_referencing_papers(self, paper_id: str) -> List[Dict]:
        """Get the papers that reference a paper."""
        self.logger.info(f'Getting referencing papers for paper {paper_id} from semantic scholar')
        referencing_papers = []
        for ref in self.get_paper_references(paper_id):
            if 'contexts' in ref:
                ref['citingPaper']['contexts'] = ref['contexts']
            referencing_papers.append(ref['citingPaper'])
        return referencing_papers

    def get_papers(self, query: List[str], domain: str, venues: List[str], min_citations: int, nr_papers: int) -> List[
        Dict]:
        collected_papers = []
        curr_query = 0
        params = {
            "query": query[0],
            "fieldsOfStudy": domain,
            "fields": "title,authors,abstract,fieldsOfStudy,venue,citationCount",
            "limit": 100,
            "offset": 0
        }
        backoff = 5
        pbar = tqdm(total=nr_papers, desc=f'Collecting papers for {domain}')

        while len(collected_papers) < nr_papers:
            try:
                rsp = requests.get('https://api.semanticscholar.org/graph/v1/paper/search', params=params,
                                   headers={'X-API-KEY': self.api_key}, timeout=self.request_timeout_sec)
                rsp.raise_for_status()
                papers = rsp.json()['data']
                papers = [paper for paper in papers if
                          paper['venue'] in venues and paper['citationCount'] >= min_citations]
                paper_ids = [paper['paperId'] for paper in papers]
                collected_len = len(collected_papers)
                collected_papers.extend(paper_ids)
                collected_papers = list(set(collected_papers))
                new_collected_len = len(collected_papers)
                pbar.update(new_collected_len - collected_len)
                params['offset'] += params['limit']
                curr_query = (curr_query + 1) % len(query)
                print(f'Curr query: {curr_query}/{len(query)}')
                if params['offset'] >= 1000:
                    curr_query = (curr_query + 1) % len(query)
                    print(f'curr_query: {curr_query}/{len(query)}')
                    params['offset'] = 0
                    params['query'] = query[curr_query]
                backoff = 5
            except requests.exceptions.RequestException as exception:
                self.logger.warning(f'Error fetching papers: {exception}')
                self.logger.info(f'Retrying in {backoff} seconds')
                time.sleep(backoff)
                backoff = min(backoff * 2, 64)
                continue

        return collected_papers[:nr_papers]
