# VIST-Inspector

This repository is devoted to VIST-Inspector, a framework that is both automatic and human-centric to evaluate visual story telling.

The file directory is organized as:

```
VIST-Inspector
└─── data
│   │   VHED_url.csv
│   └───VIST
│       │   train
│       │   val
│       │   test
│   
└─── model
    |   pre-train.ipynb
    │   ranker_single.ipynb
    │   ranker_pair.ipynb
    |   ranker_w_ling_pair.ipynb
    |   ranker_pair_multitask.py
└─── human edits analysis
    |   pre-analysis.py
    |   feature_analyze.py
``` 

## Model
`pre-train.ipynb`: implementation of stage 1, pre-training Visual-BERT with an Masked-Language-Modeling objective on VIST image-story pairs.

Files starting with "ranker" are implementation of stage 2, finetuning the pre-trained Visual-BERT on VHED dataset and adding a regression layer to output a ranking gap between a pair of stories.

`ranker_single.ipynb`: input the two stories within a story pair separately to the model to obtain their ranking scores, then subtract the score to get the ranking gap.

`ranker_pair.ipynb`: input a pair of stories together with \<SEP\> between them to the model to get the ranking gap directly.

`ranker_w_ling_pair.ipynb`: the same pairwise setting as above, but adding linguistic feature scores after story text. Here we present adding linguistic scores by directly prompting them after story text as a part of the model's input. 

`ranker_pair_multitask.ipynb`: the same pairwise setting as above, but assign a label based on which story has a higher signal score. Then we train a binary classifier that makes this prediction. The ranking loss and the signal label loss are combined as a weighted sum and backward to the whole model. Note that the code about sentence-order-prediction using ALBERT and calculating coherence score are adapted from https://github.com/usydnlp/rovist

