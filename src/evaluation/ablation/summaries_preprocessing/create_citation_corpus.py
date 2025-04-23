"""
This script collects citation information for each paper in the evaluation dataset.
"""
import argparse
import os

import pandas as pd
from util import setup_default_logger
from semantic_scholar_api import SemanticScholarAPI
from tqdm import tqdm


def collect_citations(semantic_api: SemanticScholarAPI, paper_id: str, corpus_dir: str) -> str:
    referencing_papers = semantic_api.get_referencing_papers(paper_id)
    referencing_papers_df = pd.DataFrame(referencing_papers)
    referencing_papers_df.dropna(subset=['year', 'title', 'contexts'], inplace=True)
    referencing_papers_df = referencing_papers_df[referencing_papers_df['contexts'].apply(len) > 0]
    referencing_papers_df = referencing_papers_df.reset_index(drop=True)
    referencing_papers_df['year'] = referencing_papers_df['year'].astype(int)

    referencing_papers_df = referencing_papers_df.reset_index().rename(
        columns={'index': 'id'})
    referencing_papers_df = referencing_papers_df[['id', 'title', 'year', 'contexts']]

    citations_path = os.path.join(corpus_dir, f'{paper_id}_citations.jsonl')
    referencing_papers_df.to_json(citations_path, orient='records', lines=True)

    return citations_path


def main():
    variant_dirs = os.scandir(args.eval_data_path)
    variant_dirs = [d.path for d in variant_dirs if d.is_dir() and not d.name.startswith('logs')]


    for variant_dir in variant_dirs:
        variant_dir_statements_path = os.path.join(variant_dir, 'decomposed_statements.csv')
        statements = pd.read_csv(variant_dir_statements_path)
        logger.info('Loaded %d statements from %s', len(statements), variant_dir_statements_path)

        citation_corpus_dir = os.path.join(args.output_path, 'citation_corpus')
        if not os.path.exists(citation_corpus_dir):
            os.makedirs(citation_corpus_dir)

        semantic_api = SemanticScholarAPI(args.semantic_scholar_key_path, logger, args.semantic_scholar_request_timeout_sec)

        citation_files = {}
        pbar = tqdm(total=len(statements), desc='Collecting citations...')
        for idx, statement in statements.iterrows():
            paper_id = statement['id']
            citation_file = collect_citations(semantic_api, paper_id, citation_corpus_dir)
            citation_files[paper_id] = citation_file
            pbar.update(1)

        statements['corpus_file'] = statements['id'].map(citation_files)
        variant_name = os.path.basename(variant_dir)
        if not os.path.exists(os.path.join(args.output_path, variant_name)):
            os.makedirs(os.path.join(args.output_path, variant_name))
        output_path = os.path.join(args.output_path, variant_name, 'statements_with_corpus.csv')
        statements.to_csv(output_path, index=False)
        logger.info('Wrote statements with citation files to %s', output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_data_path', type=str, required=True, help='statements to be evaluated')
    parser.add_argument('--output_path', type=str, required=True, help='output path for the preprocessed data')
    parser.add_argument('--semantic_scholar_key_path', type=str, required=True)
    parser.add_argument('--semantic_scholar_request_timeout_sec', type=int, default=10)

    args = parser.parse_args()

    logger = setup_default_logger(args.output_path)
    main()
