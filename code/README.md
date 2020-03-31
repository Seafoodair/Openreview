# How-to-run
## (1)Data anlysis
The code under 'two levels' and 'three levels' is to calculate the MJS Divergence of two levels and three levels.<br>
```two level/2017-2019.py ```the file is to claculate the MJS of two levels of 2017-2019.<br>
```three levels/2017-2019.py``` the file is to claculate the MJS of two levels of 2020.<br>
while 2020.py under 'two levels' and 'three levels' is to claculate the MJS of two/three levels of 2020.<br>
when you want to run the code,please modify the actual path of the source dir to ensure you can run the code.<br>
## (2)cluster
This module clusters abstracts and keywords to find interesting information.Please load the following command:<br>
```cd ./cluster analysis/```<br>
```python Clustering.py```<br>
Here is brief description of each file.<br>

+ "decision.txt"  Description of the reception, 1 for reception, 0 for rejection.<br>
+ "stopwords.txt" Include common early stop words in the file<br>
+ "userdict2.txt"Common key words and phrases in scientific papers<br>
## (3)sentiment analysis
### generate model<br>
The realization of emotion analysis module. Please load the following command:<br>
```cd ./sentiment analysis/```<br>
```python trainmodel.py```<br>
The script trainmodel.py first calls pretrain model to Initialization parameters.Then load the dataset training to fine tune.<br>
### sentiment predict<br>
For emotional prediction, please use the following command:python predict.py<br>
Here is brief description of each code.<br>

+ "generate.py"Match the predicted results with the review.<br>
+ "generatecount.py"Statistics of the average scores of different combinations of five angles.<br>
+ "drawsentimentpic.py"Visualization of results<br>

The format of training data is as follows:<br>
```text,label```<br>
## (4)correlation analysis
This module analyzes the relationship between paper scores and citations.Please load the following command:<br>
```cd ./citeandscore anlysis/```<br>
```python Citescatter17.py```<br>
The format of training data is as follows:<br>
id title cite final_decision average_score<br>
## (5)arxiv analysis
This module analyzes the influence of publishing on arXiv.Please load the following command:
```cd ./citeandscore anlysis/```<br>
```python ArxivGraph2017.py```<br>
If you like to crawl the raw datase,please use the following command:<br>
```python crawlarxiv.py```<br>
The script crawlarxiv.py,calls the arvix open source API port.<br>
