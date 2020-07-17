# GSRIF
This repository include code and examples for the paper "A graph-based semi-supervised reject inference framework considering imbalanced data distribution for consumer credit scoring"  

# Evironment Setting
- Python = 2.7.3
- sklearn = 0.18.2

# Code
## BM.py
BM is a traditional credit scoring model (binary classification) which is learned by accepted bad and good applicants. In other words, rejected data are not included in the BM model. Also, we do not use any imbalanced learning approach so the distribution of good and bad groups in accepted data are imbalanced as raw distribution.  

## BM-BS.py
BM-BS is another benchmark model, in which the imbalanced accepted data are over-sampled to a balanced dataset then trained as a binary classifier without any rejected data. This model is designed to test whether the improvement of classification performance is not because of reject inference but only because of imbalanced learning.

## RI-DAB.py
The RI-DAB model uses the reject inference method named “Define as bad”, i.e., the rejected samples are all labeled as bad then added into accepted data. And there is no imbalanced learning method used in it.

## RI-EXP.py
Reject inference using extrapolation. The RI-EXP model uses the reject inference method named “Extrapolation”, which assigns good-bad labels to the rejects based on the scoring model learned from accepted applicants, then a usual credit scoring model can be estimated. No imbalanced learning method is used as well.

## RI-LS.py
The RI-LS model uses the semi-supervised learning algorithm label spreading to proceed reject inference for rejected data without imbalanced learning. 

## RI-BSLS.py
RI-BSLS is our proposed novel reject inference framework for credit scoring, which uses an over-sampled accepted dataset and a randomly sampled rejected data subset for reject inference. Then a binary classification model training follows.
