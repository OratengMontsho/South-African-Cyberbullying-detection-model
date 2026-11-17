# South-African-Cyberbullying-detection-model
Transformer-Based Cyberbullying Detection for South African Multilingual Social Media Text

Code for the Honours research project Developing and Evaluating Transformer-Based Cyberbullying Detection Models for South African Multilingual Social Media Text Using Explainable AI (Sol Plaatje University).

What the study does

Three variants of AfroXLMR (Davlan/afro-xlmr-base) are trained and compared on the same South African test set of 610 isiZulu and isiXhosa records:

Model	How it is built
AfroXLMR Baseline	Fine-tuned on English cyberbullying data
AfroXLMR Adapted	Domain-adaptive pretraining on SAfriSenti, then the same English fine-tuning
AfroXLMR SA	Fine-tuned on South African data built from AfriHate

LIME and SHAP are then used to explain the predictions.

Results on the South African test set (n = 610)
Model	Accuracy	Precision	Recall	F1
AfroXLMR Baseline	48.0%	0.200	0.013	0.025
AfroXLMR Adapted	47.9%	0.259	0.023	0.042
AfroXLMR SA	84.3%	0.877	0.797	0.835
Files
cyberbullying_detection_afroxlmr.ipynb: the full pipeline, with the outputs of the runs that produced the results in the report.
Notebook sections
Setup
English cyberbullying data: balancing, 80/10/10 split and exploratory analysis
SAfriSenti corpus: record lengths and code-switching rates
Domain-adaptive pretraining (masked language modelling on SAfriSenti)
English fine-tuning of the Baseline and Adapted models
South African dataset: cleaning, duplicate removal, balancing and 80/10/10 split
AfroXLMR SA training, and evaluation of all three models
Statistical tests: confidence intervals, binomial test and McNemar's test
LIME and SHAP
Figures
Data

The datasets are not included in this repository. They are available from their original sources:

Kaggle Cyberbullying Tweets (Wang, Fu and Lu, 2020), used for English fine-tuning. https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification

SAfriSenti (Mabokela and Schlippe, 2022), used for domain-adaptive pretraining and the code-switching analysis.
https://github.com/NLPforLRLsProjects/SAfriSenti-Corpus-Mini

AfriHate, isiZulu and isiXhosa subsets (Muhammad et al., 2025), used to build the South African training, validation and test sets. https://huggingface.co/datasets/afrihate/afrihate

The notebook expects the files in a Google Drive folder at /content/drive/MyDrive/Honours/.

How to run

The notebook was run on Google Colab with a Tesla T4 GPU. Open it in Colab, mount Google Drive and run the sections in order.

Sections 4, 5 and 7 train models. Re-running them retrains the models and can change the results slightly. To check the reported results without retraining, run Section 1 and then Sections 8 to 10, which load the saved predictions and models.

Main libraries: transformers, datasets, accelerate, torch, scikit-learn, pandas, numpy, scipy, statsmodels, ftfy, wordfreq, lime, shap, matplotlib, nltk.

Content note

The datasets contain abusive and offensive language, and some of it appears in the notebook outputs.
