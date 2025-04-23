import argparse
from util import setup_default_logger
from semantic_scholar_api import SemanticScholarAPI
import json

PAPERS_CONFIG = [
    {
        "query": [
            "psychology",
            "cognitive science",
            "neuroscience",
            "behavioral science",
            "mental health",
            "cognition",
            "perception",
            "emotion",
            "decision-making",
            "psychological theory",
            "psychological models",
            "clinical psychology",
            "neuropsychology",
            "developmental psychology",
            "social psychology",
            "personality psychology",
            "psychotherapy",
            "psychopathology",
            "psychological disorders",
            "psychological experiments"
        ],
        "domain": "Psychology",
        "venues": ["The Lancet Psychiatry", "Nature Human Behaviour",
                   "American Journal of Psychiatry", "Psychological Medicine", "Journal of Affective Disorders",
                   "Neuroscience and Biobehavioral Reviews", "Translational Psychiatry", "Biological Psychiatry",
                   "Frontiers in Psychology", "American Psychologist"],
        "min_citations": 500,
        "nr_papers": 100
    },
    {
        "query": [
            "artificial intelligence",
            "machine learning",
            "deep learning",
            "natural language processing",
            "computational linguistics",
            "transformers",
            "neural networks",
            "language models",
            "AI ethics",
            "explainable AI",
            "AI fairness",
            "text classification",
            "named entity recognition",
            "sentiment analysis",
            "question answering",
            "text summarization",
            "machine translation",
            "dialogue systems",
            "large language models",
            "multimodal learning"
        ],
        "domain": "Computer Science",
        "venues": ["Association for Computational Linguistics",
                   "Empirical Methods in Natural Language Processing",
                   "North American Chapter of ACL",
                   "European Chapter of ACL",
                   "Conference on Neural Information Processing Systems",
                   "International Conference on Learning Representations",
                   "International Conference on Machine Learning"],
        "min_citations": 500,
        "nr_papers": 100
    },
    {
        "query": [
            "medicine",
            "medical research",
            "healthcare",
            "biomedical science",
            "clinical research",
            "public health",
            "pathology",
            "diagnostics",
            "medical imaging",
            "patient care",
            "oncology",
            "cardiology",
            "neurology",
            "infectious diseases",
            "mental health",
            "drug discovery",
            "precision medicine",
            "genomics",
            "biotechnology",
            "epidemiology"
        ],
        "domain": "Medicine",
        "venues": ["The Lancet", "New England Journal of Medicine", "New England Journal of Medicine",
                   "Nature Medicine", "The Lancet Oncology", "Journal of Clinical Oncology",
                   "Journal of the American Medical Association", "European Heart Journal", "The Lancet Neurology",
                   "Circulation", "Journal of the American College of Cardiology"],
        "min_citations": 500,
        "nr_papers": 100
    }
]


def main():
    semantic_api = SemanticScholarAPI(args.semantic_scholar_api_key_path, logger,
                                      args.semantic_scholar_request_timeout_sec)
    results = {}
    for conf in PAPERS_CONFIG:
        papers = semantic_api.get_papers(query=conf['query'], domain=conf['domain'], venues=conf['venues'],
                                         min_citations=conf['min_citations'], nr_papers=conf['nr_papers'])
        results[conf['domain']] = papers

    config_path = f'{args.output_path}/config.json'
    with open(config_path, 'w') as f:
        json.dump(PAPERS_CONFIG, f)
    logger.info(f'Saved config to {config_path}')

    out_path = f'{args.output_path}/papers.json'
    with open(out_path, 'w') as f:
        json.dump(results, f)

    logger.info(f'Saved papers to {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, required=True, help='output path')
    parser.add_argument('--semantic_scholar_api_key_path', type=str, required=True,
                        help='Semantic Scholar API key path')
    parser.add_argument('--semantic_scholar_request_timeout_sec', type=int, default=10, help='timeout for requests')

    args = parser.parse_args()
    logger = setup_default_logger(args.output_path)

    main()
