import argparse
import json
import os
import ast
import openai
import time
import pandas as pd
from util import setup_default_logger, request_openai_batch_completions, get_openai_batch_completions, encode_sources, \
    get_claim_start_end_years, cluster_sources, get_sources_textual_representation, get_field_of_study, \
    unique_field_of_study
from sentence_transformers import SentenceTransformer, util


def main():
    variant_dirs = os.scandir(args.eval_data_path)
    variant_dirs = [d.path for d in variant_dirs if d.is_dir() and '=' in d.name]
    fields_of_study = get_field_of_study(args.test_set_path)

    verification_prompt_template = open(args.verification_prompt_path).read()
    sentence_encoder = SentenceTransformer(args.sentence_encoder).to('cuda')
    openai_client = openai.OpenAI(api_key=open(args.openai_key_path).read().strip())

    batches = {}

    for variant_dir in variant_dirs:
        variant_statements_path = os.path.join(variant_dir, 'statements_with_corpus.csv')
        logger.info(f'Loading data from {variant_statements_path}')

        eval_data = pd.read_csv(variant_statements_path, dtype={'decomposed_statements': object})
        eval_data['decomposed_statements'] = eval_data['decomposed_statements'].apply(ast.literal_eval)
        logger.info(f'Loaded evaluation data from {args.eval_data_path}')

        prompts = {}
        sources_by_statement_id = {}
        bad_period = 0
        nr_claims = {field: 0 for field in unique_field_of_study}
        for _, statement in eval_data.iterrows():
            statement_field = fields_of_study[statement['id']]
            prompt_template = verification_prompt_template.replace('{{PAPER_NAME}}', statement['title'])
            corpus_filename = os.path.basename(statement['corpus_file'])
            corpus_file_path = os.path.join(args.eval_data_path, 'citation_corpus', corpus_filename)
            logger.info(f'Loading corpus from {corpus_file_path}')
            corpus = pd.read_json(corpus_file_path, orient='records', lines=True)
            corpus = encode_sources(sentence_encoder, corpus)
            decomposed_statements = statement['decomposed_statements']['impact']

            sources_by_statement_id[statement['id']] = {}
            for claim in decomposed_statements:
                nr_claims[statement_field] += 1
                claim_sources = corpus
                if args.temporal_retrieval and 'impact_period' in claim:
                    start_year, end_year = get_claim_start_end_years(claim['impact_period'])
                    if start_year and end_year:
                        claim_sources = [source for source in claim_sources if
                                         source['year'] and start_year <= source['year'] <= end_year]
                    else:
                        bad_period += 1
                claim_sources = cluster_sources(claim_sources, args.cluster_threshold)

                if 'impact_description' not in claim:
                    logger.warning(f'Claim {claim} does not have an impact description, skipping...')
                    continue

                prompt = prompt_template.replace('{{CLAIM}}', claim["impact_description"])
                claim_sources_text = get_sources_textual_representation(claim_sources)
                sources_by_statement_id[statement['id']][claim['id']] = claim_sources_text

                prompt = prompt.replace('{{SOURCES}}', claim_sources_text)

                prompts[f'{statement["id"]}_{claim["id"]}'] = prompt

        eval_data['verification_sources'] = eval_data['id'].map(sources_by_statement_id)
        output_path = os.path.join(variant_dir, 'statements_with_verification_sources.csv')
        eval_data.to_csv(output_path, index=False)
        logger.info(f'Running faithfulness evaluation with {len(prompts)} prompts')
        batch_id = request_openai_batch_completions(prompts, max_tokens=1024, temperature=0.0, batch_idx=0,
                                                    output_path=args.output_path, engine=args.openai_engine,
                                                    client=openai_client)

        batches[batch_id] = {'variant_dir': variant_dir, 'updated_eval_data_path': output_path,
                             'bad_period': bad_period}
        nr_calims_path = os.path.join(variant_dir, 'nr_claims.json')
        with open(nr_calims_path, 'w') as f:
            json.dump(nr_claims, f)

    logger.info(f'Submitted all batch requests to OpenAI:\n{json.dumps(batches, indent=2)}')
    batches_path = os.path.join(args.output_path, 'openai_batches.json')
    with open(batches_path, 'w') as f:
        json.dump(batches, f)
    logger.info(f'Wrote batch info to {batches_path}')

    logger.info('Waiting for completions...')

    finished_batches = {batch_id: False for batch_id in batches.keys()}
    while not all(finished_batches.values()):
        for batch_id, finished in finished_batches.items():
            if finished:
                continue
            response_by_id, batch_status = get_openai_batch_completions(batch_id, openai_client)
            if batch_status == 'completed':
                logger.info(f'Batch {batch_id} completed!')
                finished_batches[batch_id] = True
                batches[batch_id]['responses'] = response_by_id
                batch_results_path = os.path.join(args.output_path, f'{batch_id}_results.json')
                with open(batch_results_path, 'w') as f:
                    json.dump(response_by_id, f)
                logger.info(f'Wrote batch results to {batch_results_path}')
            else:
                logger.info(f'Batch {batch_id} status: {batch_status}')
        time.sleep(60)

    logger.info('All batches completed!')

    all_variant_results = pd.DataFrame()
    for batch_id, batch_info in batches.items():
        eval_data = pd.read_csv(batch_info['updated_eval_data_path'])
        verified_claims = {field: 0 for field in unique_field_of_study}
        response_by_id = batch_info['responses']
        responses_by_statement_id = {}
        for response_id, response in response_by_id.items():
            statement_id, claim_id = response_id.split('_')
            if statement_id not in responses_by_statement_id:
                responses_by_statement_id[statement_id] = {}
            field = fields_of_study[statement_id]
            responses_by_statement_id[statement_id][claim_id] = response

            response_start = response.find('<answer>')
            response_end = response.find('</answer>')
            if response_start != -1 and response_end != -1:
                response = response[response_start + len('<answer>'):response_end].strip()
                if response.lower() == 'yes':
                    verified_claims[field] += 1
            elif '**Answer:**' in response:
                response = response.split('**Answer:**')[1].split('\n')[0].strip()
                if response.lower() == 'yes':
                    verified_claims[field] += 1
            else:
                logger.warning(f'No answer found in response: {response}')

        eval_data['claim_verification_results'] = eval_data['id'].map(responses_by_statement_id)
        output_path = os.path.join(batch_info['variant_dir'], 'statements_with_verification_responses.csv')
        eval_data.to_csv(output_path, index=False)

        nr_claims = json.load(open(os.path.join(batch_info['variant_dir'], 'nr_claims.json')))
        variant_name = os.path.basename(batch_info['variant_dir'])
        run_name = 'run1'
        if 'run1_' in variant_name:
            variant_name = variant_name.split('run1_')[1]
        if 'run2_' in variant_name:
            variant_name = variant_name.split('run2_')[1]
            run_name = 'run2'

        variant_results = {
            'variant': variant_name,
            'run': run_name,
        }

        for field in unique_field_of_study:
            variant_results[f'nr_verified_claims_{field}'] = verified_claims[field]
            variant_results[f'nr_claims_{field}'] = nr_claims[field]
            variant_results[f'faithfulness_{field}'] = verified_claims[field] / nr_claims[field]

        variant_results['total_verified_claims'] = sum(verified_claims.values())
        variant_results['total_claims'] = sum(nr_claims.values())
        variant_results['total_faithfulness'] = variant_results['total_verified_claims'] / variant_results[
            'total_claims']

        all_variant_results = pd.concat([all_variant_results, pd.DataFrame([variant_results])])
        logger.info(f'Variant results: {json.dumps(variant_results, indent=2)}')

    output_path = os.path.join(args.output_path, 'faithfulness_results.csv')
    all_variant_results.to_csv(output_path, index=False)
    logger.info(f'Wrote faithfulness results to {output_path}')

    results_str = all_variant_results.to_csv(sep='\t', index=False)
    logger.info(f'Faithfulness results:\n{results_str}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_data_path', type=str, required=True, help='statements to be evaluated')
    parser.add_argument('--output_path', type=str, required=True, help='output path')
    parser.add_argument('--verification_prompt_path', type=str, required=True, help='verification prompt path')
    parser.add_argument('--openai_engine', type=str, required=True, help='OpenAI engine')
    parser.add_argument('--openai_key_path', type=str, required=True, help='OpenAI key path')
    parser.add_argument('--sentence_encoder', type=str, help='Used to compute similarity between claims and sources')
    parser.add_argument('--cluster_threshold', type=float, help='Threshold for clustering sources')
    parser.add_argument('--temporal_retrieval', action='store_true', help='Use temporal retrieval to select sources',
                        default=True)
    parser.add_argument('--test_set_path', type=str)

    args = parser.parse_args()

    logger = setup_default_logger(args.output_path)
    main()
