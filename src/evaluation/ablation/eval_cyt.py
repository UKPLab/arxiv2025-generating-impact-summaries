import argparse
import json
import os

from util import setup_default_logger, get_claim_start_end_years, year_is_in_time_period, parse_citations, \
    parse_evidence, get_field_of_study, unique_field_of_study
import pandas as pd
import ast
from tqdm import tqdm


def main():
    variant_dirs = os.scandir(args.data_path)
    variant_dirs = [d.path for d in variant_dirs if d.is_dir() and '=' in d.name and 'citations=none' not in d.name]
    logger.info(f'Found variant directories: {variant_dirs}')
    papers_field_of_study = get_field_of_study(args.test_set_path)

    all_variants_results = pd.DataFrame()

    for variant_dir in variant_dirs:
        logstr = ""
        field_hist = {field: 0 for field in unique_field_of_study}
        variant_statements_path = os.path.join(variant_dir, 'statements_with_corpus.csv')
        logger.info(f'Loading data from {variant_statements_path}')
        eval_data = pd.read_csv(variant_statements_path, dtype={'decomposed_statements': object})
        eval_data['decomposed_statements'] = eval_data['decomposed_statements'].apply(ast.literal_eval)

        nr_real_evidence = {field: 0 for field in unique_field_of_study}
        nr_period_matching_evidence = {field: 0 for field in unique_field_of_study}
        all_evidence = {field: 0 for field in unique_field_of_study}
        pbar = tqdm(total=len(eval_data))
        for _, statement in eval_data.iterrows():
            logstr += f'=========={statement["title"]}==========\n'
            statement_field = papers_field_of_study[statement['id']]
            field_hist[statement_field] += 1
            statement_citations_text = statement['citations']
            statement_citations = parse_citations(statement_citations_text)
            decomposed_statement = statement['decomposed_statements']['impact']
            for impact_desc in decomposed_statement:
                raw_evidence = impact_desc.get('evidence', [])
                impact_period_key = 'impact_period' if 'impact_period' in impact_desc else 'impact_periods'
                start_year, end_year = get_claim_start_end_years(impact_desc.get(impact_period_key, ''))

                logstr += f'======\n[{start_year} - {end_year}] {impact_desc["aspect_of_period"]}\n'
                if 'impact_description' in impact_desc:
                    logstr += f'{impact_desc["impact_description"]}\n---\n'
                evidence = parse_evidence(raw_evidence)
                if not evidence:
                    logger.warning(
                        f'Failed to parse evidence for {raw_evidence}, impact_desc: {impact_desc}')
                    logstr += '----Evidence\n'
                    logstr += f'Failed to parse evidence for {raw_evidence}, impact_desc: {impact_desc}\n'

                all_evidence[statement_field] += len(evidence)
                if not start_year or not end_year:
                    logger.info(f'{impact_desc.get(impact_period_key, "")} is not a valid period')
                    continue
                for e in evidence:
                    if e in statement_citations:
                        nr_real_evidence[statement_field] += 1
                        if year_is_in_time_period(statement_citations[e]['year'], (start_year, end_year)):
                            nr_period_matching_evidence[statement_field] += 1
                            logstr += '#SUCCESS!\n'
                            logstr += json.dumps(statement_citations[e], indent=2) + '\n'
                        else:
                            logstr += '#FAILURE!\n'
                            logstr += json.dumps(statement_citations[e], indent=2) + '\n'
            pbar.update(1)

        sanity_log_path = os.path.join(args.output_path, 'sanity_check.log')
        with open(sanity_log_path, 'w') as f:
            f.write(logstr)
        logger.info(f'Wrote sanity check log to {sanity_log_path}')

        logger.info(f'Processed {len(eval_data)} statements')
        variant_name = os.path.basename(variant_dir)
        run_name = 'run1'
        if 'run1' in variant_name:
            variant_name = variant_name.replace('run1_', '')
        elif 'run2' in variant_name:
            variant_name = variant_name.replace('run2_', '')
            run_name = 'run2'

        variant_results = {
            'variant': variant_name,
            'run': run_name
        }
        for field in unique_field_of_study:
            variant_results[f'nr_real_evidence_{field}'] = nr_real_evidence[field]
            variant_results[f'nr_period_matching_evidence_{field}'] = nr_period_matching_evidence[field]
            variant_results[f'all_evidence_{field}'] = all_evidence[field]
            variant_results[f'period_matching_evidence_ratio_{field}'] = nr_period_matching_evidence[field] / \
                                                                         all_evidence[field]
            variant_results[f'real_evidence_ratio_{field}'] = nr_real_evidence[field] / all_evidence[field]

        variant_results['total_real_evidence'] = sum(nr_real_evidence.values())
        variant_results['total_period_matching_evidence'] = sum(nr_period_matching_evidence.values())
        variant_results['total_all_evidence'] = sum(all_evidence.values())
        variant_results['total_period_matching_evidence_ratio'] = variant_results['total_period_matching_evidence'] / \
                                                                  variant_results['total_all_evidence']
        variant_results['total_real_evidence_ratio'] = variant_results['total_real_evidence'] / \
                                                       variant_results['total_all_evidence']

        all_variants_results = pd.concat([all_variants_results, pd.DataFrame([variant_results])])

        logger.info(f'Field of study distribution: {field_hist}')

    output_path = os.path.join(args.output_path, 'evidence_eval_results.csv')
    all_variants_results.to_csv(output_path, index=False)
    logger.info(f'Wrote evidence evaluation results to {output_path}')

    results_text = all_variants_results.to_csv(sep='\t', index=False)
    logger.info(f'Evaluation results:\n{results_text}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--test_set_path', type=str)
    args = parser.parse_args()
    logger = setup_default_logger(args.output_path)
    main()
