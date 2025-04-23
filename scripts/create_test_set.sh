#!/bin/bash -x
#SBATCH --time=168:00:00
#SBATCH -c1
#SBATCH --mem-per-cpu=10g
#SBATCH --mail-user=noy.sternlicht@mail.huji.ac.il
#SBATCH --mail-type=ALL

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

python3 src/collect_papers.py \
  --output_path "collect_papers_results" \
  --semantic_scholar_api_key_path "secret_keys/semantic_scholar.txt" \
  --semantic_scholar_request_timeout_sec 15
