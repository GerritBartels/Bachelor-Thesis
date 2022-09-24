# About this Repository

This repository contains the code to our (Jacob Dudek and Gerrit Bartels) bachelor thesis. We implemented 5 Neural Networks and compared their performance on the task of unconditional text generation. We set up a thorough evaluation scheme using common automatic evaluation methods and supplemented these results with a human survey. Furthermore, we propose two evaluation metrics based on the Jennsen-Shannon Distance that help with judging how well the underlying data distribution has been learnt. 

---
# Dataset

As dataset for training all models we used a monolingual news crawl from the Fifth Conference on Machine Translation (WMT20) that can directly be obtained from the conference [website](https://data.statmt.org/news-crawl/en/}{data.statmt.org/news-crawl/en/). It contains approximately 44M English news sentences extracted from online newspaper articles that were published throughout 2019. After applying our preprocessing steps (see preprocessing notebook) we obtained a dataset comprising approximately 240k sentences with an average length of 18.65 and a vocabulary size of 6801.

---
# Models

## LSTMLM
<img src="https://github.com/GerritBartels/Bachelor-Thesis/blob/main/Architecture%20Visualizations/LSTMLM.png" width="500" />

## GSGAN
<img src="https://github.com/GerritBartels/Bachelor-Thesis/blob/main/Architecture%20Visualizations/GSGAN.png" width="500" />

## GPT-2 Small
<img src="https://github.com/GerritBartels/Bachelor-Thesis/blob/main/Architecture%20Visualizations/GPT2_Small.png" height="500" />

## cVAELM
<img src="https://github.com/GerritBartels/Bachelor-Thesis/blob/main/Architecture%20Visualizations/cVAELM.png" width="500" />

## LaTextGAN
<img src="https://github.com/GerritBartels/Bachelor-Thesis/blob/main/Architecture%20Visualizations/LatextGAN.png" width="500" />

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

|                                 |  LSTM<br>LM |  cVAE<br>LM  |   GS<br>GAN  | LaText<br>GAN | GPT-2<br>Small | Real <br>Data |
|:-------------------------------:|:-----------:|:------------:|:------------:|:-------------:|:--------------:|:-------------:|
|    **Average<br>Sent Length**   | **_16.83_** |    _12.91_   |    _16.89_   |    _18.06_    |     _17.25_    |    _16.66_    |
| **JS Distance <br>Sent Length** |   _0.1471_  |   _0.4334_   |   _0.1677_   |    _0.3487_   |  **_0.1206_**  |    _0.0205_   |
| **JS Distance<br>Token Counts** |   _0.1441_  |   _0.2963_   | **_0.1437_** |    _0.5111_   |    _0.2444_    |    _0.1286_   |
| **Top 12<br>Token Overlap**     | **_12/12_** |    _10/12_   |  **_12/12_** |     _7/12_    |     _11/12_    |    _12/12_    |
| **Test BLEU-4**                 |   _0.3136_  |   _0.0544_   |   _0.3258_   |    _0.2563_   |  **_0.4536_**  |    _0.3301_   |
| **Self BLEU-4**                 |   _0.3235_  | **_0.0904_** |   _0.3463_   |    _0.6746_   |    _0.5374_    |    _0.3282_   |
| **FID**                         |   _0.369_   |    0.9932    | **_0.3606_** |    _1.9926_   |    _0.7368_    |    _0.3456_   |

Results of the automatic evaluation methods applied to all models and, for reference, to the test data itself. The best results are highlighted in **bold**.

|                               | LSTM<br>LM | cVAE<br>LM | GS<br>GAN | LaText<br>GAN | GPT-2<br>Small | Real <br>Data |
|:-----------------------------:|:----------:|:----------:|:---------:|:-------------:|:--------------:|:-------------:|
| **Average<br>Fluency Rating** |  _3.0704_  |  _1.9296_  |  _3.1861_ |    _1.704_    |  **_3.9025_**  |    _4.3948_   |
|   **Confusion Rate<br>in %**  |   _23.81_  |   _9.93_   |  _20.37_  |     _9.03_    |   **_29.82_**  |      _-_      |

Results of the human evaluation applied to all models and, for reference, to the
test data itself. The best results are highlighted in **bold**.

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
