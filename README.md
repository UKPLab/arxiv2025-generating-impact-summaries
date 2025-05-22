[![Arxiv](https://img.shields.io/badge/Arxiv-YYMM.NNNNN-red?style=flat-square&logo=arxiv&logoColor=white)](https://put-here-your-paper.com)
[![License](https://img.shields.io/github/license/UKPLab/generating-impact-summaries)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/badge/Python-3.11-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)



<p align="center">
  <img src="impact_summaries_icon.png" alt="Centered Image" width="90" />
</p>

# In-depth Research Impact Summarization through Fine-Grained Temporal Citation Analysis




Understanding the impact of scientific publications is crucial for identifying breakthroughs and
                        guiding future research. Traditional metrics like citation counts often miss the nuanced ways a
                        paper contributes to its field. In this work, we propose a new task: generating nuanced,
                        expressive, and time-aware impact summaries that capture both praise (confirmation citations)
                        and critique (correction citations) through the evolution of fine-grained citation intents. We
                        introduce an evaluation framework tailored to this task, showing moderate to strong human
                        correlation on subjective metrics such as insightfulness. Expert feedback from professors
                        reveals strong interest in these summaries and suggests future improvements.

This repo contains the code and data used to produce the experiments in this paper. Namely: 1) to generate fine-grained intents of citation contexts and 2) generate impact summaries based on the classified intents.


Contact person: [Hiba Arnaout](mailto:hiba.arnaout@tu-darmstadt.de) 

[UKP Lab](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt](https://www.tu-darmstadt.de/
)


### Getting Started
#### Prerequisites
* Python 3.11.2 or higher
* Some code requires an OpenAI API key.
* Some code requires a GPU.

#### Installation

```bash
# Recommended: Create and activate a virtual environment
python3 -m venv myenv
source ./myenv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Setting up the OpenAI API
Some experiments require an OpenAI API key. You can set it up by following the instructions [here](https://beta.openai.com/docs/developer-quickstart/).
After you have the API key, create a simple text file `secret_keys/openai.txt` and paste the key there. The code will automatically read the key from this file.


### Reproducing Results
This part describe how to reproduce the results presented in our the paper.

#### Ablation

```bash
# Decompress the preprocessed variant results.
tar -xf data/preprocessed_summaries.tar.xz -C data/preprocessed_summaries

# Run the ablation experiments
chmod +x scripts/run_ablation_experiments.sh
./scripts/run_ablation_experiments.sh
```

You can rerun the data preprocessing by executing `scripts/ablation_summaries_preprocess.sh`. Note this requires a [Semantic Scholar API key](https://www.semanticscholar.org/product/api#api-key-form)

### Citation
If you use this code or data in your research, please cite our paper:

```bibtex
@misc{arnaout2025indepthresearchimpactsummarization,
      title={In-depth Research Impact Summarization through Fine-Grained Temporal Citation Analysis}, 
      author={Hiba Arnaout and Noy Sternlicht and Tom Hope and Iryna Gurevych},
      year={2025},
      eprint={2505.14838},
      archivePrefix={arXiv},
      primaryClass={cs.DL},
      url={https://arxiv.org/abs/2505.14838}, 
}
```

### Authors

Hiba Arnaout, Noy Sternlicht, Tom Hope, Iryna Gurevych.


### Disclaimer

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 
