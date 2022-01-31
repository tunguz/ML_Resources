
# ML Resources

GitHub Repo with various ML/AI/DS resources that I find useful. I'll populate it with links to articles, libraries, and other resources that I come across. Hoping for more or less regular, ongoing updates.


## General ML Resources

* How to avoid machine learning pitfalls: a guide for academic researcher: https://arxiv.org/pdf/2108.02497.pdf


## Tabular Data ML

### General

* Tabular Data: Deep Learning is Not All You Need: https://arxiv.org/abs/2106.03253
* Declarative Machine Learning Systems: https://arxiv.org/abs/2107.08148

### NNs for Tabular Data

* pytorch-widedeep, deep learning for tabular data IV: Deep Learning vs LightGBM. A thorough comparison between DL algorithms and LightGBM for tabular data for classification and regression problems: https://jrzaurin.github.io/infinitoml/2021/05/28/pytorch-widedeep_iv.html

* SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training: https://github.com/somepago/saint (repo)
* SAINT paper: https://arxiv.org/abs/2106.01342
* Regularization is all you Need: Simple Neural Nets can Excel on Tabular Data https://arxiv.org/abs/2106.11189
* Revisiting Deep Learning Models for Tabular Data: https://arxiv.org/abs/2106.11959v1
* TaBERT: Pretraining for Joint Understanding of Textual and Tabular Data: https://arxiv.org/abs/2005.08314v1
* Gradient Boosting Neural Networks: GrowNet. Paper: https://arxiv.org/abs/2002.07971, Code: https://github.com/sbadirli/GrowNet

### Boosting

* Januschowski, Tim, et al. "Forecasting with trees." International Journal of Forecasting (2021)
https://www.sciencedirect.com/science/article/pii/S0169207021001679
* XGboost documentation: https://xgboost.readthedocs.io/en/latest/

### Neural Networks - General

* Every Model Learned by Gradient Descent Is Approximately a Kernel Machine: https://arxiv.org/abs/2012.00152

### NNs vs Kernel Methods

* The quest for adaptivity: exploring https://francisbach.com/quest-for-adaptivity/

### Ensembling

* Automatic Frankensteining: Creating Complex Ensembles Autonomously: https://epubs.siam.org/doi/abs/10.1137/1.9781611974973.83

### Stacking

* vecstack is a handy little library that implements the stacking transformations with your train and test data. It has both the functional interface and the sklearn fit transform interface: https://github.com/vecxoz/vecstack

### Autoencoders

* RecoTour III: Variational Autoencoders for Collaborative Filtering with Mxnet and Pytorch: https://jrzaurin.github.io/infinitoml/2020/05/15/mult-vae.html
* Michael Jahrer's famous Porto Seguro Kaggle competition solution: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629
* Kaggle Tabular Playground Series February 2021 - 1st place solution writeup: https://www.kaggle.com/c/tabular-playground-series-feb-2021/discussion/222745


### Shapeley Values

* The Shapley Taylor Interaction Index: https://arxiv.org/abs/1902.05622

### Adversarila Validation

* Adversarial Validation Approach to Concept Drift Problem in User Targeting Automation Systems at Uber: https://arxiv.org/abs/2004.03045


## Auto ML

### Books and Articles

* Automated Machine Learning: Methods, Systems, Challenges. Probably the single best monograph on AutoML. Published in 2019, so not quite the cutting edge, but still very useful. https://www.amazon.com/Automated-Machine-Learning-Challenges-Springer-ebook/dp/B07S3MLGFW/
* Towards Automated Machine Learning: Evaluation and Comparison of AutoML Approaches and Tools: https://arxiv.org/abs/1908.05557
* AutoML: A Survey of the State-of-the-Art: https://arxiv.org/abs/1908.00709
* Can AutoML outperform humans? An evaluation on popular OpenML datasets using AutoML Benchmark: https://arxiv.org/abs/2009.01564

### Auto ML Software

* H2O Driverless AI documentation: https://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/index.html

### Hyperparameter Tuning Software

* Hyperopt: https://github.com/hyperopt/hyperopt
* hyperopt-sklearn: https://github.com/hyperopt/hyperopt-sklearn
* Optuna: https://optuna.readthedocs.io/en/stable/
* Uber Turbo: https://github.com/uber-research/TuRBO


## Computer Vision

* An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale: https://arxiv.org/abs/2010.11929
* Scaling Vision Transformers: ‘As a result, we successfully train a ViT model with two billion parameters, which attains a new state-of-the-art on ImageNet of 90.45% top-1 accuracy. The model also performs well on few-shot learning, for example, attaining 84.86% top-1 accuracy on ImageNet with only 10 examples per class.’
https://arxiv.org/abs/2106.04560
* Revisiting ResNets: Improved Training and Scaling Strategies: https://arxiv.org/abs/2103.07579
* Diffusion Models Beat GANs on Image Synthesis: https://arxiv.org/abs/2105.05233


## NLP

* Evaluating Large Language Models Trained on Code: https://arxiv.org/abs/2107.03374
* How exactly does word 2 vec work? https://www.semanticscholar.org/paper/How-exactly-does-word-2-vec-work-Meyer/49edbe35390224dc0c19aefe4eb28312e70b7e79
* Attention Is All You Need: https://arxiv.org/abs/1706.03762
* Big Bird: Transformers for Longer Sequences: https://arxiv.org/abs/2007.14062
* Fast WordPiece Tokenization: https://arxiv.org/abs/2012.15524
* A Recipe For Arbitrary Text Style Transfer with Large Language Models: https://arxiv.org/abs/2109.03910
* How much do language models copy from their training data? Evaluating linguistic novelty in text generation using RAVEN: https://arxiv.org/abs/2111.09509
* Transformers from Scratch: https://e2eml.school/transformers.html
* Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model: https://arxiv.org/abs/2201.11990

## RecSys

* Boosting algorithms for a session-based, context-aware recommender system in an online travel domain: https://doi-org.stanford.idm.oclc.org/10.1145/3359555.3359557
* How Sensitive is Recommendation Systems’ Offline Evaluation to Popularity? https://core.ac.uk/download/pdf/296221513.pdf

## ML applications in Natural Sciences

* Skilful precipitation nowcasting using deep generative models of radar: https://www.nature.com/articles/s41586-021-03854-z
* Predictive models of RNA degradation through dual crowdsourcing: https://arxiv.org/abs/2110.07531


## Social Media Analysis

* Birds of the Same Feather Tweet Together. Bayesian Ideal Point Estimation Using Twitter Data. Analysis of homophily of Politicians on Twitter. http://pablobarbera.com/static/barbera_twitter_ideal_points.pdf
* Leadership Communication and Power: Measuring Leadership in the U.S. House of Representatives from Social Media Data: https://preprints.apsanet.org/engage/apsa/article-details/60c239b28214c646e0a61589

## Policy

* Final Report: National Security Commission on Artificial Intelligence https://www.nscai.gov/wp-content/uploads/2021/03/Full-Report-Digital-1.pdf

## Mathematics

* Exploring the beauty of pure mathematics in novel ways https://deepmind.com/blog/article/exploring-the-beauty-of-pure-mathematics-in-novel-ways

## Bilogy

* AlphaFold: a solution to a 50-year-old grand challenge in biology: https://deepmind.com/blog/article/alphafold-a-solution-to-a-50-year-old-grand-challenge-in-biology

## ML Competition Platforms

* Kaggle: https://www.kaggle.com
* Eval AI: https://eval.ai
* Zindi: https://zindi.africa
* Driven Data: https://www.drivendata.org
* Codalab: https://competitions.codalab.org/

# Other Resources

## Knowledge Graphs

* NodePiece: Compositional and Parameter-Efficient Representations of Large Knowledge Graphs: https://arxiv.org/abs/2106.12144v1

## Statistics

* Statistical Modeling: The Two Cultures (with comments and a rejoinder by the author): https://projecteuclid.org/journals/statistical-science/volume-16/issue-3/Statistical-Modeling--The-Two-Cultures-with-comments-and-a/10.1214/ss/1009213726.full

## Quantum Computing

TensorFlow Quantum: A Software Framework for Quantum Machine Learning: https://arxiv.org/abs/2003.02989


## Installation Guides

* Install CUDA 11.2, cuDNN 8.1.0, PyTorch v1.8.0 (or v1.9.0), and python 3.9 on RTX3090 for deep learning https://medium.com/analytics-vidhya/install-cuda-11-2-cudnn-8-1-0-and-python-3-9-on-rtx3090-for-deep-learning-fcf96c95f7a1

## Self drivign vehicles

* Autonomy 2.0: Why is self-driving always 5 years away? https://arxiv.org/abs/2107.08142v1

![competition](https://road-to-kaggle-grandmaster.vercel.app/api/badges/tunguz/competition)
![dataset](https://road-to-kaggle-grandmaster.vercel.app/api/badges/tunguz/dataset)
![notebook](https://road-to-kaggle-grandmaster.vercel.app/api/badges/tunguz/notebook)
![discussion](https://road-to-kaggle-grandmaster.vercel.app/api/badges/tunguz/discussion)
