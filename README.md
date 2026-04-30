# Archelec topic modeling project

This repository contains a clean Python pipeline for the NLP course project on the Archelec corpus.

## Project idea

We compare classical topic modeling methods (LDA and NMF) with BERTopic on French electoral manifestos. We then analyze the relation between extracted topics and candidate metadata (`titulaire-soutien`, `titulaire-profession`, `titulaire-age-tranche`).

This version uses **standard BERTopic without KMeans**. It adds Archelec-specific stopwords and removes repeated archive boilerplate such as `Sciences Po / fonds CEVIPOF` from cleaned text.

## Data setup

From the project root:

```powershell
cd data\raw
git clone https://gitlab.teklia.com/ckermorvant/arkindex_archelec
cd arkindex_archelec
Get-ChildItem -Path .\text_files -Recurse -Filter *.zip | ForEach-Object {
    Expand-Archive -Path $_.FullName -DestinationPath $_.DirectoryName -Force
}
cd ..\..\..
```

Put the metadata and stopwords here:

```text
data/raw/archelect_search.csv
data/raw/stop_word_fr.txt
```

## Run

From the project root:

```powershell
python -m scripts.run_01_prepare_data
python -m scripts.run_02_baselines
python -m scripts.run_03_bertopic
```

Then open:

```text
outputs/topics/manual_topic_labels.csv
```

Fill the `manual_label` column manually. This file intentionally contains no example text column.

Then run:

```powershell
python -m scripts.run_04_analysis
```

## Outputs

Main outputs:

```text
outputs/topics/baseline_topics.csv
outputs/topics/bertopic_topics.csv
outputs/topics/manual_topic_labels.csv
outputs/topics/document_topics.csv
outputs/evaluation/topic_model_evaluation.csv
outputs/metadata/metadata_association_tests.csv
outputs/metadata/prediction_results.csv
outputs/figures/topic_size_distribution.png
outputs/figures/topic_by_support_heatmap.png
outputs/figures/topic_by_profession_heatmap.png
outputs/figures/topic_by_age_heatmap.png
```

## Notes

- LDA and NMF are classical baselines.
- BERTopic is the transformer-based comparison model.
- If BERTopic remains noisy, it is acceptable to use NMF as the main interpretable model and BERTopic as a neural comparison.
- The report should discuss this honestly rather than forcing BERTopic to look better.
