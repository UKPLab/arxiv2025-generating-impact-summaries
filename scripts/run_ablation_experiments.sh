#!/bin/bash -x
#SBATCH --time=30:00:00
#SBATCH --gres gg4:g
#SBATCH --mem-per-cpu=10g

echo $PWD
activate() {
  . $PWD/myenv/bin/activate
}

set_env_vars() {
  PYTHONPATH=$PWD/src
  export PYTHONPATH

  HF_DATASETS_CACHE=$PWD/.datasets_cache
  export HF_DATASETS_CACHE

  HF_HOME=$PWD/.hf_home
  export HF_HOME
}

activate
set_env_vars

module load cuda
module load nvidia

#------------------------TRUSTWORTHINESS
#---CYT
python3 src/evaluation/ablation/eval_cyt.py --data_path "data/preprocessed_summaries" \
  --output_path "cyt_results" \
  --test_set_path "data/test_set.tsv"

#---Faithfulness
python3 src/evaluation/ablation/eval_faithfulness.py \
  --eval_data_path "data/preprocessed_summaries" \
  --output_path "faithfulness_eval_out" \
  --verification_prompt_path "src/evaluation/ablation/verification_prompt.txt" \
  --openai_engine "gpt-4o-mini" \
  --openai_key_path "secret_keys/openai.txt" \
  --sentence_encoder "all-mpnet-base-v2" \
  --cluster_threshold 0.05 \
  --temporal_retrieval \
  --test_set_path "data/test_set.tsv"
