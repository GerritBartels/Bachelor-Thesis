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
To assess the capabilities of our models in unconditional text generation, we employed methods that evaluate the model outputs with respect to sentence diversity (D) and quality (Q), as well as how well the underlying data distribution was captured (C). We also conducted a survey to get an additional angle at judging model performance.

## Automatic Measures:
* JS Distance Sentence Lengths (C)
* JS Distance Token Counts (C)
* Test BLEU-4 (Q)
* Self BLEU-4 (D)
* Fréchet InferSent Distance (Q & D)

## Human Evaluation
The survey was implemented in [\_magpie](https://magpie-ea.github.io/magpie-site/), a minimal architecture for generating portable interactive experiments and made available as a web-based online survey through the hosting service Netlify. We defined two tasks to elicit judgments about the overall quality of generation (Likert-scale rating) and the participants' likelihood of detecting whether a sentence was artificially generated (2-alternative forced-choice task).

---
# Results


---
# Example Sentences

## LSTMLM
* elizabeth warren suggests trump would win the u.s through congress, whereas president trump by his <NUM> th year race as he staged a century.
* unable to read live in recent times, china is not long term.
* please note that radio <NUM> had a site of panic and pre recorded surveillance books in the afternoon little body.
* and in san diego that may have been trumps remarks after a bitter short tournament.
* government employees, women and organisations have been focused on improving care and role to ensure guests be held responsible for their personal data.

## GSGAN
* should multi billion dollar corporations zero emissions by <NUM>?
* the mother of a girl next to her was pushed too hard.
* labour responded that they should not vote by the snp, then we would need to get brexit done.
* but another west london, royal republic, won <NUM> european international womens semi finals
* our future brexit will turn us once again, he said during his three day visit.

## GPT-2 Small
* when the new government started being introduced in october, there was no such thing as a result that could ever take place.
* some lawmakers are going to move forward in the next phase of the senate in a week, as congress does.
* she said: it did not feel right and i did not want this to be happening at all.
* however, he was left with a six year old who left with the job over £ <NUM>.

## cVAELM
* ministers way.
* twitter will you fell aside an additional public supply chain of women if
* nothing every divided on february on me on, but we for.
* it once certainly normally neither this all their scores remain on.
* professional annually.

## LaTextGAN
* thirds he need kong he rt.com lanka rt.com lanka rt.com lanka rt.com lanka rt.com lanka rt.com lanka rt.com lanka rt.com lanka rt.com lanka rt.com lanka rt.com lanka rt.com lanka
* dow is not united do well in europe, but will be an interview.
* rt.com rt.com just angeles rt.com feel angeles thrones am angeles have knew the people, in that <NUM> am not on <NUM> <NUM>.
* there two do in trump and an emergency and go on an emergency services to <NUM> %.
* $ president a need to <NUM> and no deal to climate change u.s border on monday.
