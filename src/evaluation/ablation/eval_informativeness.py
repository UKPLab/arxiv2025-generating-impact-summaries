import argparse
import os
import requests
import pandas as pd
import ast
import time
from tqdm import tqdm


logger = open('after_logger.txt', 'w')

from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
unit_tests = {"Insightfulness": 1, "Trend Awareness": 1, "Specificity": 1}

#metrics
insight_metric = GEval(
    name="Insightfulness",
    evaluation_steps=[
      "Determine whether the impact summary describes how the paper has been directly used by or influenced by other works ." ,
      "Assess how well the impact summary articulates the paper ’s influence with informative details ." ,
      "You should heavily penalize the impact summary for lack of insight ."
        ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
)
trend_metric = GEval(
    name="Trend awareness",
    evaluation_steps=[
      "Determine whether the impact summary mentions how the impact of the paper has changed over time , ensuring each impact period is clearly identified with descriptive titles ." ,
      "You should heavily penalize the impact summary if the titles of consecutive impact periods are not diverse ."
      ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
)
specificity_metric = GEval(
    name="Specificity",
    evaluation_steps=[
      "Determine whether the impact summary mentions specific techniques , frameworks , or studies influenced by the paper , or if it remains broad and lacking supporting details." ,
      "You should heavily penalize the impact summary if it only restates the title and abstract or provides vague , generic statements without concrete examples of the influence of the paper."
          ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
)


def get_full_statement(decomposed_statement, paper_name):
    statement = f"**{paper_name}**\n\n"
    for impact_desc in decomposed_statement:
        period_key = 'impact_period' if 'impact_period' in impact_desc else 'impact_periods'
        if period_key in impact_desc and 'aspect_of_period' in impact_desc:
            statement += f"*{impact_desc[period_key]}: {impact_desc['aspect_of_period']}*\n"
        if 'impact_description' in impact_desc:
            statement += f"{impact_desc['impact_description']}\n\n"

    return statement



variant_dirs = os.scandir('/variants_results/json_files')
variant_dirs = [d.path for d in variant_dirs if d.is_dir() and '=' in d.name]
logger.write(f'Found variant directories: {variant_dirs}')
logger.write('\n')

all_variant_results = pd.DataFrame()

for variant_dir in variant_dirs:
  variant_dir_statements_path = os.path.join(variant_dir, 'statements_with_corpus.csv')
  logger.write(f'Loaded evaluation data from %s {variant_dir_statements_path}')
  logger.write('\n')
  eval_data = pd.read_csv(variant_dir_statements_path, dtype={'decomposed_statements': object})
  eval_data['decomposed_statements'] = eval_data['decomposed_statements'].apply(ast.literal_eval)
  logger.write(f'Variant name {variant_dir}')
  logger.write('\n')
  quality_test_results = {}
  pbar = tqdm(total=len(eval_data))
  for _, statement in eval_data.iterrows():
      decomposed_statement = statement['decomposed_statements']['impact']
      paper_name = statement['title']
      logger.write(f'Paper name {paper_name}')
      logger.write('\n')
      statement_text = get_full_statement(decomposed_statement, paper_name)

      statement_results = {}
      for name, something in unit_tests.items():
          statement_results[name] = -1
          test_case = LLMTestCase(input="What is the scientific impact of the paper titled \"{{PAPER_NAME}}\"?".replace("{{PAPER_NAME}}", paper_name), actual_output=statement_text)
          logger.write(f'unit_test {name}')
          logger.write('\n')
          score = 0
          time.sleep(1.5)
          print(statement_text)
          if name == 'Insightfulness':
            insight_metric.measure(test_case)
            score = insight_metric.score
            logger.write(f'Score {insight_metric.score}')
            logger.write('\n')
            logger.write(f'Reason {insight_metric.reason}')
            logger.write('\n')
          if name == 'Trend Awareness':
            trend_metric.measure(test_case)
            score = trend_metric.score
            logger.write(f'Score {trend_metric.score}')
            logger.write('\n')
            logger.write(f'Reason {trend_metric.reason}')
            logger.write('\n')
          if name == 'Specificity':
            specificity_metric.measure(test_case)
            score = specificity_metric.score
            logger.write(f'Score {specificity_metric.score}')
            logger.write('\n')
            logger.write(f'Reason {specificity_metric.reason}')
            logger.write('\n')

          statement_results[name] = score
      statement_results['statement'] = statement_text

      quality_test_results[statement['id']] = statement_results
      pbar.update(1)

  mean_scores = {}
  for unit_test_name in unit_tests:
      mean_scores[unit_test_name] = sum(
          [v[unit_test_name] for v in quality_test_results.values() if unit_test_name != 'statement']) / len(
          quality_test_results)


  results_df = pd.DataFrame(mean_scores.items(), columns=['unit_test', 'mean_score'])
  output_path = os.path.join(variant_dir, 'after_quality_test_results.csv')
  results_df.to_csv(output_path, index=False)
  logger.write(f'Wrote quality test results to {output_path}')
  logger.write('\n')

  statements_results_df = pd.DataFrame.from_dict(quality_test_results, orient='index')
  print(statements_results_df)
  statements_output_path = os.path.join(variant_dir, 'after_quality_test_statement_results.csv')
  statements_results_df.to_csv(statements_output_path)
  logger.write(f'Wrote quality test statement results to {statements_output_path}')
  logger.write('\n')

  variant_name = os.path.basename(variant_dir)
  run_name = 'run1'
  if 'run1' in variant_name:
      variant_name = variant_name.replace('run1_', '')
  if 'run2' in variant_name:
      variant_name = variant_name.replace('run2_', '')
      run_name = 'run2'

  results_df['variant'] = variant_name
  results_df['run'] = run_name
  all_variant_results = pd.concat([all_variant_results, results_df])

  results_df_text = results_df.to_csv(sep='\t', index=False)
  logger.write(f'Quality test results:\n{results_df_text}')
  logger.write('\n')

all_variant_results_output_path = os.path.join('/variants_results/', 'after_quality_test_results.csv')
all_variant_results.to_csv(all_variant_results_output_path, index=False)
logger.write(f'Wrote all quality test results to {all_variant_results_output_path}')
logger.write('\n')

all_variant_results_text = all_variant_results.to_csv(sep='\t', index=False)
logger.write(f'All quality test results:\n{all_variant_results_text}')
logger.write('\n')

logger.close()
