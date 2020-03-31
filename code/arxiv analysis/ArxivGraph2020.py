#coding=utf-8
"""

date:Feb,17,2020

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#name_list = ['2016-04', '2017-04', '2017-05','2017-06','2017-07','2017-09','2017-11','2017-12','2018-01\n submission','2018-02','2018-03','2018-04\n notification','2018-05','2018-06','2018-07\n conference','2018-08','2018-09','2018-10',
             #'2018-11','2018-12','2019-01','2019-02','2019-03','2019-04']
name_list = [ '18\\10','18\\11','18\\12','19\\01','19\\02','19\\03','19\\04','19\\05','19\\06','19\\07','19\\08',
             '19\\09\n sub','19\\10','19\\11','19\\12\n nt','20\\01','20\\02']
num_list = [ 3,2,2,5,1,2,4,8,13,5,6,23,18,13,23,7,8]
num_list2 = [ 2,5,0,6,5,2,7,39,34,18,13,39,51,39,33,13,8]
#num_list = [0, 1, 2, 0, 0,0,4,1,0,5,6,3,12,14,12,7,18,42,13,15,21,20,10,11]
#num_list2 = [1, 0, 1, 1, 2, 1,0,2,3,4,6,7,27,14,8,13,18,42,16,20,20,2,3,5]
x = list(range(len(num_list)))

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 32,
}
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 5,
}

total_width, n = 0.8, 2
width = total_width / n
plt.bar(x, num_list, width=width, tick_label=name_list,label='accept', fc='green')#accept
for i in range(len(x)):
    x[i] += width
plt.bar(x, num_list2, width=width, label='reject',  fc='red')#reject
#plt.title('2019ICLR')
plt.ylabel('Number of papers',font1)
plt.xlabel('Date',font1)
#plt.tick_params(labelsize=13,rotation=12)
#plt.tick_params(labelsize=15)
plt.yticks(fontsize=25)
plt.xticks(fontsize=20,rotation=60)
plt.legend(prop=font1)
plt.show()