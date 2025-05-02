## Files in this directory
* `extended_PST_dataset.csv`: A large-scale groundtruth dataset for classifying citation context into "impact-revealing" citations or "other" (incidental citations). The dataset contains ~70k citation contexts from 105 papers. The data contains the following fields:
  * `id`: Data-entry ID
  * `citing_id_PST`: PST ID of the citing paper.
  * `citing_id_Semantic_scholar`: Semantic Scholar ID of the citing paper.
  * `citing_title`: The title of the citing paper.
  * `cited_id_PST`:  PST ID of the cited paper.
  * `cited_title`: The title of the cited paper.
  * `context`: A list of sentences from the citing paper containing the citation.
  * `intent_class`: Either "other" or "impact-revealing".
  * `fine-grained_intent`: A more nuanced description of the citation intent, for example "reporting prior findings and applications".
* `finegrained_intents.zip`: The generated fine-grained intents for ~70k citation contexts using our method. The fields are similar to those of `extended_PST_dataset.csv`.
* `outputs.tar.xz`: The generated impact summaries for 105 papers (using different variants of our method).
* `preprocessed_summaries.tar.xz`: Pre-processed summaries (from the outputs.zip file) used for evaluation.
* `test_set.tsv`: The 105 papers used in our experiments.
* `training_examples_k=50.csv`: The 50 training examples used in our "identifying impact-revealing citations" task (ICL setting).
