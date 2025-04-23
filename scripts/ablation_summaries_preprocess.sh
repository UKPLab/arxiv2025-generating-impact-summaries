#!/bin/bash -x
#SBATCH --time=168:00:00
#SBATCH -c1
#SBATCH --mem-per-cpu=10g

echo $PWD
activate() {
  . $PWD/myenv/bin/activate
}

set_env_vars() {
  PYTHONPATH=$PWD/src
  export PYTHONPATH
}

activate
set_env_vars

python3 src/evaluation/ablation/summaries_preprocessing/decompose_statements.py \
  --eval_data_path "data/variants_output" \
  --output_path "data/preprocessed_summaries"



python3 src/evaluation/ablation/summaries_preprocessing/create_citation_corpus.py \
  --eval_data_path "data/preprocessed_summaries" \
  --output_path "data/preprocessed_summaries" \
  --semantic_scholar_key_path "secret_keys/semantic_scholar.txt" \
  --semantic_scholar_request_timeout_sec 15
