import numpy as np
import sys
import torch
import array
import math
import pandas as pd
from d2lsource.GetMySQLConnection import GetMySQLConnection
from scipy.stats import entropy as H
def get_score_2017_2018_2020(text):
    index=text.index('Rating:###')
    index1=int(index)+len('Rating:###')
    str=text[index1:len(text)]
    index2=str.index(':')+index1

    return text[index1:index2]

def JSD(prob_distributions, weights, logbase=2):
    # left term: entropy of mixture
    wprobs = weights * prob_distributions
    mixture = wprobs.sum(axis=0)
    entropy_of_mixture = H(mixture, base=logbase)

    # right term: sum of entropies
    entropies = np.array([H(P_i, base=logbase) for P_i in prob_distributions])
    wentropies = weights * entropies
    # wentropies = np.dot(weights, entropies)
    sum_of_entropies = wentropies.sum()

    divergence = entropy_of_mixture - sum_of_entropies
    return(divergence)
import scipy.stats
def JS_divergence(p,q,r):
    M=(p+q+r)/3
    JS_1=0.5*scipy.stats.entropy(p, M)+0.5*scipy.stats.entropy(q, M)
    JS_2=0.5*scipy.stats.entropy(p, M)+0.5*scipy.stats.entropy(r, M)
    JS_3=0.5*scipy.stats.entropy(q, M)+0.5*scipy.stats.entropy(r, M)

    return (JS_1+JS_2+JS_3)/3
def get_dis(list1,list2,list3):
    total=[]
    for line in list1:
        if line not in total:
            total.append(line)
    for line in list2:
        if line not in total:
            total.append(line)
    for line in list3:
        if line not in total:
            total.append(line)
    import numpy as np
    return_list1=np.zeros(len(total))
    return_list2=np.zeros(len(total))
    return_list3=np.zeros(len(total))
    for line in list1:
        return_list1[total.index(line)]+=1
    for line in list2:
        return_list2[total.index(line)]+=1
    for line in list3:
        return_list3[total.index(line)]+=1
    return return_list1/np.sum(return_list1),return_list2/np.sum(return_list2),return_list3/np.sum(return_list3)


import math
def get_log(num):

    return float(math.log(num,2))

def get_review_level(text):
    if text=='Experience Assessment:###I do not know much about this area.':
        return '1'
    elif text=='Experience Assessment:###I have read many papers in this area.':
        return '2'
    elif text=='Experience Assessment:###I have published one or two papers in this area.':
        return '3'
    elif text=='Experience Assessment:###I have published in this field for several years.':
        return '4'
    else:
        print("error")

# connection=GetMySQLConnection('localhost' , 'root' , 'root' , 'openreviewuseddata',3307)
# original_data_2020=connection.selectDb("select  A,L,M from tp_2020conference")

import pandas as pd
import os
data=pd.read_excel(os.path.abspath('../data/tp_2020conference.xlsx'),sheet_name='tp_2020conference')

original_data_2020=data[['title','reviewer_level','paper_score']].values.tolist()

title_2020=[original_data_2020[0][0]]
for line in original_data_2020:
    if line[0] not in title_2020:
        title_2020.append(line[0])
print(len(title_2020))

situations_list=[
    [1, 2, 3],
    [1, 2, 4],
    [1, 3, 4],
    [2, 3, 4]
]




for situation in situations_list:
    MJS=[]
    sample_one = []
    sample_two = []
    sample_three=[]
    total = []
    counter=0
    sum_total=0

    for line in title_2020:
        distribution_before_original_1 = []
        distribution_before_original_2 = []
        distribution_before_original_3=[]

        for single_data in original_data_2020:
            if line==single_data[0]:
                if int(get_review_level(single_data[1]))==situation[0]:
                    distribution_before_original_1.append(int(get_score_2017_2018_2020(single_data[2])))
                elif int(get_review_level(single_data[1]))==situation[1]:
                    distribution_before_original_2.append(int(get_score_2017_2018_2020(single_data[2])))
                elif int(get_review_level(single_data[1]))==situation[2]:
                    distribution_before_original_3.append(int(get_score_2017_2018_2020(single_data[2])))
        if len(distribution_before_original_1)>0 and len(distribution_before_original_2)>0 and len(distribution_before_original_3)>0:
            counter += 1
            sum_total += len(distribution_before_original_1)
            sum_total += len(distribution_before_original_2)
            sum_total += len(distribution_before_original_3)
            avg_1 = np.mean([int(i) for i in distribution_before_original_1])
            avg_2 = np.mean([int(i) for i in distribution_before_original_2])
            avg_3 = np.mean([int(i) for i in distribution_before_original_3])

            avg_total = np.mean([int(i) for i in
                                 distribution_before_original_1 + distribution_before_original_2 + distribution_before_original_3])

            # for line in distribution_before_original_1:
            MJS.append(avg_1 * get_log(avg_1 / avg_total))
            # for line in distribution_before_original_2:
            MJS.append(avg_2 * get_log(avg_2 / avg_total))
            # for line in distribution_before_original_3:
            MJS.append(avg_3 * get_log(avg_3 / avg_total))
        # total.append([i for i in list(set(sample_one))])
    if len(MJS) > 0:
        print(situation)
        print(np.sum(MJS) / (3 * counter))
        print(counter)
        print(sum_total)
        print("###########")
    else:
        print(situation)
        print("not exists")
        print("###########")
























