# About this Repository

This repository contains the code to our (Jacob Dudek and Gerrit Bartels) bachelor thesis. We implemented 5 Neural Networks and compared their performance on the task of unconditional text generation. We set up a thorough evaluation scheme using common automatic evaluation methods and supplemented these results with a human survey. Furthermore, we propose two evaluation metrics based on the Jennsen-Shannon Distance that help with judging how well the underlying data distribution has been learnt. 

---
# The Dataset

As dataset for training all models we used a monolingual news crawl from the Fifth Conference on Machine Translation (WMT20) that can directly be obtained from the conference [website](https://data.statmt.org/news-crawl/en/}{data.statmt.org/news-crawl/en/). It contains approximately 44M English news sentences extracted from online newspaper articles that were published throughout 2019. After applying our preprocessing steps (see preprocessing notebook) we obtained a dataset comprising approximately 240k sentences with an average length of 18.65 and a vocabulary size of 6801.

---
# The Models

## LSTMLM

## GSGAN
![GSGAN Architecture](https://github.com/GerritBartels/Bachelor-Thesis/blob/main/Architecture%20Visualizations/LSTMLM_transparent.png?raw=true)

## GPT-2 Small

## cVAELM

## LaTextGAN


---
# Evaluation Methods
To assess the capabilities of our models in unconditional text generation, we employed methods that evaluate the model outputs with respect to sentence diversity and quality, as well as how well the underlying data distribution was captured. We also conducted a survey to get an additional angle at judging model performance.

The survey was implemented in [\_magpie](https://magpie-ea.github.io/magpie-site/), a minimal architecture for generating portable interactive experiments and made available as a web-based online survey through the hosting service Netlify. We defined two tasks to elicit judgments about the overall quality of generation (Likert-scale rating) and the participants' likelihood of detecting whether a sentence was artificially generated (2-alternative forced-choice task).

---
# Results


---
# Example Sentences
