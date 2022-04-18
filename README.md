# Openreview [中文](https://mp.weixin.qq.com/s/doBnqcdaP4xEGYY3kDR2eQ)
ICLR open review dataset
## The ICLR Open Reviews dataset<br>
1.ICLR data is a dataset of scientific peer reviews available to help researchers study this important artifact. The dataset consists of over 10K paper informatiron and the corresponding accept/reject decisions in top-tier venues from ICLR conference, as well as over 40K textual peer reviews written by reviewer for a review of the papers. For details of data introduction and format, please see [./data/README.md](./data/README.md)<br>
2.This experiment mainly uses three parts of data, including ICLR data, arXiv data and cite data. Other data are extracted or spliced from these data.<br>
3.Our experiments mainly include data acquisition, data analysis, clustering analysis, correlation analysis, emotional analysis.For details, please see [./code/README.md](./code/README.md)<br>
## Setup Configuration
```
pip install -r requirements.txt
pip install tensorflow==1.15.0
```
## Acknowledgement
1.We thank the openreview.net, arxiv.org for their commitment to promoting transparency and openness in scientific communication.<br>
2.We used the ELECTRA pre-training model. https://github.com/google-research/electra <br>
3.This work was supported by the National Natural Science Foundation of China (62072082, U1811261), and the Key R&D Program of Liaoning Province
(2020JH 2/10100037).
