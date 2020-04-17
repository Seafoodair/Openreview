# Openreview
ICLR open review dataset
## Corrigendum
In the section 3.2,we draw the Fig5.Due to carelessness, the y-axis coordinates are reversed.The revised Fig5 is as follows:<br>
![ad](https://github.com/Seafoodair/Openreview/blob/master/img-folder/sentiment.png)
 We can see that higher review score often comes with more positive aspects from a macro perspective, which is under expectation. We observe that most of reviews with score higher than 6 do NOT have negative comments on presentation and motivation, but may allow some flaws on relate work, experiment, and novelty. The reviewers that have overall positive to the paper are likely to pose improvement suggestions on motivation, experiment, and presentation to make the paper perfect. Novelty quality and experiment seem to be mentioned
more frequently than the other aspects, and positive sentiment on novelty
is distributed more unevenly from high-score reviews to low-score reviews. This
implies that novelty does play important role in making the decision.
It is also interesting that there is no review in which all aspects are positive or
negative. It is unlikely that a paper is perfect in all aspects or has no merit. Reviewers are also likely to be more rigorous in good papers and be more tolerant
with poor papers.
## The ICLR Open Reviews dataset<br>
1.ICLR data is a dataset of scientific peer reviews available to help researchers study this important artifact.The dataset consists of over 5K paper informatiron and the corresponding accept/reject decisions in top-tier venues from ICLR conference, as well as over 20K textual peer reviews written by reviewer for a review of the papers.For details of data introduction and format,please see [./data/README.md](./data/README.md)<br>
2.This experiment mainly uses three parts of data, including ICLR data, arXiv data and cite data. Other data are extracted or spliced from these data.<br>
3.Our experiments mainly include data acquisition, data analysis, clustering analysis, correlation analysis, emotional analysis.For details, please see [./code/README.md](./code/README.md)<br>
## Setup Configuration
```
pip install -r requirements.txt
pip install tensorflow==1.15.0
```
## Acknowledgement
1.We thank the openreview.net, arxiv.org for their commitment to promoting transparency and openness in scientific communication.<br>
2.We used the ELECTRA pre-training model.https://github.com/google-research/electra
