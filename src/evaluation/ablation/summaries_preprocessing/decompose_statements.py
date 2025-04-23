import argparse
import json
import os.path
import uuid

import pandas as pd
from util import setup_default_logger, request_openai_batch_completions, get_openai_batch_completions


def parse_statement(statement: str):
    statement = statement.replace('```json', '').replace('```', '').strip()
    try:
        statement = json.loads(statement)
    except json.JSONDecodeError:
        logger.info('Failed to parse statement: %s', statement)
        statement = {}

    if 'impact_periods' in statement and isinstance(statement['impact_periods'], list):
        if len(statement['impact_periods']) == 0:
            logger.info('Statement contains empty impact periods list: %s', statement)
        return statement
    else:
        logger.info('Statement does not contain impact periods: %s', statement)

    return {}


def main():
    data_files = os.listdir(args.eval_data_path)
    data_files = [os.path.join(args.eval_data_path, f) for f in data_files if f.endswith('.jsonl')]

    shared_ids = set()
    for data_file in data_files:
        data = pd.read_json(data_file, orient='records', lines=True)
        shared_ids = shared_ids.intersection(set(data['id'])) if shared_ids else set(data['id'])
    logger.info('Loaded %d shared IDs from %d data files', len(shared_ids), len(data_files))

    for data_file in data_files:
        eval_data = pd.read_json(data_file, orient='records', lines=True)
        eval_data = eval_data[eval_data['id'].isin(shared_ids)]
        logger.info('Loaded %d statements from %s', len(eval_data), data_file)

        variant_name = os.path.basename(data_file).replace('.jsonl', '')
        variant_out_dir = os.path.join(args.output_path, variant_name)
        if not os.path.exists(variant_out_dir):
            logger.info('Creating output directory %s', variant_out_dir)
            os.makedirs(variant_out_dir)

        statements = {}
        total_nr_impact_claims = 0
        for idx, row in eval_data.iterrows():
            statement = row['response']
            statement = parse_statement(statement)
            impact_claims = []
            for impact_claim_info in statement.get('impact_periods', []):
                impact_claim_info['id'] = str(uuid.uuid4())
                impact_claims.append(impact_claim_info)
                total_nr_impact_claims += 1

            statement['impact'] = impact_claims
            statements[row['id']] = statement

        logger.info('Decomposed %d impact claims from %d statements', total_nr_impact_claims, len(statements))
        decomposed_statements = statements
        eval_data['decomposed_statements'] = eval_data['id'].map(decomposed_statements)
        output_path = os.path.join(variant_out_dir, 'decomposed_statements.csv')
        eval_data.to_csv(output_path, index=False)
        logger.info('Wrote %d decomposed statements to %s', len(eval_data), output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_data_path', type=str, required=True, help='statements to be evaluated')
    parser.add_argument('--output_path', type=str, required=True, help='output path for the preprocessed data')

    args = parser.parse_args()
    logger = setup_default_logger(args.output_path)

    main()
