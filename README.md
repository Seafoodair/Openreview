# Openreview
Data and code for Gang Wang et al., ECML-PKDD 2020's paper titled "What Have We Learned from OpenReview?"<br>
## The ICLR Open Reviews dataset<br>
1.ICLR data is a dataset of scientific peer reviews available to help researchers study this important artifact.The dataset consists of over 5K paper informatiron and the corresponding accept/reject decisions in top-tier venues from ICLR conference, as well as over 20K textual peer reviews written by reviewer for a review of the papers.<br>
2.This experiment mainly uses three parts of data, including ICLR data, arXiv data and cite data. Other data are extracted or spliced from these data.<br>
3.Our experiments mainly include data acquisition, data analysis, clustering analysis, correlation analysis, emotional analysis.For details, please see [./code/README.md](./code/README.md)<br>
## Setup Configuration
```
pip install -r requirements.txt
```
## Acknowledgement
1.We thank the openreview.net, arxiv.org for their commitment to promoting transparency and openness in scientific communication.<br>
2.We used the ELECTRA pre-training model.https://github.com/google-research/electra
