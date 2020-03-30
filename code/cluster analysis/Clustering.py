import os
import nltk
from nltk.corpus import stopwords
import sys
import sys
from nltk.tokenize import MWETokenizer
from d2lsource.GetMySQLConnection import GetMySQLConnection

sys.setrecursionlimit(100000) 

path = '../2020TXTSET/'
titles = []
files = []
counter=0
for filename in os.listdir(path):
    titles.append(filename)
    
    filestr = open(path + filename, encoding='utf-8_sig').read()
    files.append(filestr)
    counter=counter+1

def remove_symbols(sentence):
    import string
    del_estr = string.punctuation + string.digits  # ASCII 标点符号，数字
    replace = " " * len(del_estr)
    tran_tab = str.maketrans(del_estr, replace)
    sentence = sentence.translate(tran_tab)
    return sentence
# 对句子进行分词
def segment(text, userdict_filepath="userdict2.txt", stopwords_filepath='stopwords.txt'):
    import nltk
    # nltk..load_userdict(userdict_filepath)
    stopwords = [line.strip().lower() for line in open(stopwords_filepath, 'r', encoding='utf-8').readlines()]  # 这里加载停用词的路径
    # seg_list = nltk.word_tokenize(text.replace(',',' '))

    final_list = []
    temp_list = []
    with open(userdict_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            temp_list.append(line.strip(' ').strip('\n'))
    f.close()
    temp = []
    for line in temp_list:
        for li in line.lower().split(' '):
            if len(li) != 0:
                temp.append(li.strip('\t'))
        final_list.append(tuple(temp))
        temp.clear()

    userdict_list = final_list
    # print(userdict_list)
    tokenizer = MWETokenizer(userdict_list, separator=' ')

    seg_list=tokenizer.tokenize(nltk.word_tokenize(remove_symbols(text).lower()))

    seg_list_without_stopwords = []

    for word in seg_list:
        if word not in stopwords:
            if word != '\t':
                seg_list_without_stopwords.append(word)
    return seg_list_without_stopwords

print(segment('natural*language(process@basis)feature+Vectors'))

totalvocab_tokenized = []
for i in files:
    allwords_tokenized = segment(i, "userdict2.txt", 'stopwords.txt')
    totalvocab_tokenized.extend(allwords_tokenized)

final_vocabulary=list(set(totalvocab_tokenized))
print(len(totalvocab_tokenized))
print(len(final_vocabulary))
print("&&&&&&&&&&&&&")
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_features=200000,
                                  stop_words='english',
                                 use_idf=True, tokenizer=segment)
# print(len(tfidf_vectorizer))

# print(tfidf_vectorizer)

tfidf_matrix = tfidf_vectorizer.fit_transform(files) #fit the vectorizer to synopses
print(tfidf_matrix.shape)

# for line in tfidf_matrix.toarray():
#     # print(line)
#     for li in line:
#         if  li >0:
#             print(li)
#     print(len(line))
#     break

from sklearn.metrics.pairwise import cosine_similarity

dist=1-cosine_similarity(tfidf_matrix)
print(dist.shape)
for line in dist:
    print(line)
import sys
sys.exit(0)
from scipy.cluster.hierarchy import ward, dendrogram, linkage,set_link_color_palette
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei'] #用来正常显示中文标签
#Perform Ward's linkage on a condensed distance matrix.
#linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances
#Method 'ward' requires the distance metric to be Euclidean
# linkage_matrix = linkage(dist, method='average', metric='euclidean', optimal_ordering=False)
# linkage_matrix = linkage(dist, method='average', metric='euclidean', optimal_ordering=False)
linkage_matrix = linkage(dist, method='ward', metric='euclidean', optimal_ordering=False)
# linkage_matrix = linkage(dist, method='average', metric='euclidean', optimal_ordering=False)
#Z[i] will tell us which clusters were merged, let's take a look at the first two points that were merged
#We can see that ach row of the resulting array has the format [idx1, idx2, dist, sample_count]
# print(linkage_matrix)

#获取聚类矩阵
# temp_string=''
# with open('linkage_matrix_result.txt','a+',encoding='utf-8') as f:
#     for line in linkage_matrix:
#         for li in line:
#             if isinstance(li,float):
#                 temp_string=temp_string+"%.6f"%li+' '
#             else:
#                 temp_string = temp_string +str(li) + ' '
#         temp_string.strip(' ')
#         f.write(temp_string+'\n')
#         temp_string=''
# f.close()
# sys.exit(0)
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)
    with open('temp_result.txt','a+',encoding='utf-8') as f:
        for key1,key2,key3,key4,key5 in zip(ddata['icoord'],ddata['dcoord'],ddata['ivl'],ddata['leaves'],ddata['color_list']):
            # f.write(str(key1)+'\n')
            # f.write(str(key2)+'\n')
            # f.write(str(key3)+'\n')
            # f.write(str(key4)+'\n')
            f.write(str(key5)+'\n')
            # f.write('*************\n')
    f.close()


    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                print(x)
                print(y)
                print('***************')
                plt.plot(x, y, 'o', c=c)
                plt.annotate("", (x, y), xytext=(0, -5), textcoords='offset points', va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata


from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

c, coph_dists = cophenet(linkage_matrix, pdist(dist))

print("*****************")

print(c)

print("*******************")


import matplotlib.pyplot as plt

#根据录取情况制定颜色
path = 'D:/FilesForBERT/BERTProject/SimilarBetweenSentences/ArrangementClustering/2020TXTSET/'
titles = []
files = []
counter=0
for filename in os.listdir(path):
    titles.append("————————"+str(filename).lower())

# print(titles)

symbol_list=[]

for line in titles:
    index = line.find('##')  # 第一次出现的位置
    index2 = line.find('##', index + 1)  # 第二次出现的位置
    # print(line[index+2:index2])
    symbol_list.append(line[index+2:index2])

dict={}

for i,j in zip(titles,symbol_list):
    # print(i+"&&&&"+j)
    dict[i]=j

print(dict)

D_leaf_colors = {}

# for key in dict.keys():
#     if dict[key]=='1':
#         D_leaf_colors[key[0:18]]='#008000'
#     else:
#         D_leaf_colors[key[0:18]] = '#FF0000'
#
# print(D_leaf_colors)
# link_cols = {}
# for i, i12 in enumerate(linkage_matrix[:,:2].astype(int)):
#   c1, c2 = (link_cols[x] if x > len(linkage_matrix) else D_leaf_colors["attr_%d"%x]for x in i12)
#   link_cols[i+1+len(linkage_matrix)] = c1 if c1 == c2 else '#008000'
import scipy.cluster.hierarchy as sc

# plt.figure(figsize=(75, 30))
temp_title=[]
for line in titles:
    temp_title.append(line)

from scipy.cluster.hierarchy import fcluster
max_d = 5.0
plt.figure(figsize=(30, 10))

cluster_id = fcluster(linkage_matrix, max_d, criterion='distance')

with open('cluster_result.txt','a+',encoding='utf-8') as f:
    for line in cluster_id:
        f.write(str(line)+'\n')
f.close()

print("cluster_id")
print(len(list(set(cluster_id))))

# sys.exit(0)


decision=[]
with open('decision.txt','r',encoding='utf-8') as f:

    for line in f:
        decision.append(int(line.strip('\n')))

f.close()
print(len(decision))


dict_total={}
dict_accept={}
list_list=[]
for i in range(0,len(list(set(cluster_id))),1):
  list_list.append([])
# print(list_list)

accept_rate=[]
for i in range(1, len(list(set(cluster_id))) + 1, 1):
    dict_total[str(i)] = 0
    dict_accept[str(i)] = 0

for i, j in zip(cluster_id, decision):
    if j == 1:
        dict_accept[str(i)] = dict_accept[str(i)] + 1
    dict_total[str(i)] = dict_total[str(i)] + 1
print(dict_accept)
print(dict_total)
for key1, key2 in zip(dict_total.keys(), dict_accept.keys()):
    if dict_total[key1] != 0:
        accept_rate.append(dict_accept[key2] / dict_total[key1])
    else:
        accept_rate.append(0)
counter_pointer=0

print(accept_rate)

def Normalization(x):
  return [(float(i) - min(x)) / float(max(x) - min(x)) for i in x]

distance_max=Normalization(accept_rate)

def color(value):
  digit = list(map(str, range(10))) + list("ABCDEF")
  if isinstance(value, tuple):
    string = '#'
    for i in value:
      a1 = i // 16
      a2 = i % 16
      string += digit[a1] + digit[a2]
    return string
  elif isinstance(value, str):
    a1 = digit.index(value[1]) * 16 + digit.index(value[2])
    a2 = digit.index(value[3]) * 16 + digit.index(value[4])
    a3 = digit.index(value[5]) * 16 + digit.index(value[6])
    return (a1, a2, a3)

for key in dict.keys():
    D_leaf_colors[key]=str(color((255-int(distance_max[cluster_id[titles.index(key)]-1]*255),int(distance_max[cluster_id[titles.index(key)]-1]*255),0)))

print(D_leaf_colors)

fancy_dendrogram(linkage_matrix,labels=temp_title,   leaf_rotation=-90., leaf_font_size=12., show_contracted=True, annotate_above=10, max_d=max_d)

label_colors=D_leaf_colors

ax = plt.gca()
xlbls = ax.get_xmajorticklabels()

print(len(xlbls))
from d2lsource.GetMySQLConnection import GetMySQLConnection
import sys
connection=GetMySQLConnection('localhost' , 'root' , 'root' , 'openreviewuseddata',3307)
sql_list=[]
temp_sql=""
for i in range(len(xlbls)):
    xlbls[i].set_color(label_colors[xlbls[i].get_text()])
    temp_sql="insert into tp_cluster_temp_result values ('%s','%s');"%(str(xlbls[i].get_text()),str(label_colors[xlbls[i].get_text()]))
    sql_list.append(temp_sql)
    # temp_sql=""

for line in sql_list:
    connection.insertDB(line)
connection.closeDb()

plt.show()


