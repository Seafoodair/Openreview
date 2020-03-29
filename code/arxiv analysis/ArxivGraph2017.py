#coding=utf-8
"""

date：Feb,17,2020

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
name_list = ['15\\10', '15\\11', '16\\02','16\\03','16\\04','16\\05','16\\06','16\\07','16\\08','16\\09','16\\10','16\\11\n submission','16\\12','17\\01','17\\02\n notification','17\\03','17\\04\n conference']
num_list = [1, 0, 1, 4, 0, 5,2,0,3,7,6,43,11,4,13,7,5]
num_list2 = [0, 1, 0, 2, 2, 3,2,1,0,2,10,27,4,6,4,0,2]
x = list(range(len(num_list)))
"""
total_width, n = 0.4, 2
width = total_width / n
plt.bar(x, num_list, width=width, label='accept', fc='green')#accept
for i in range(len(x)):
    x[i] += width
plt.bar(x, num_list2, width=width, label='reject', tick_label=name_list, fc='red')#reject
plt.title('2017ICLR')
plt.ylabel('Number')
plt.xlabel('Date')
plt.legend()
plt.show()
"""
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 32,
}

total_width, n = 0.8, 2
width = total_width / n
plt.bar(x, num_list, width=width, label='accept', fc='green')#accept
for i in range(len(x)):
    x[i] += width
plt.bar(x, num_list2, width=width, label='reject', tick_label=name_list, fc='red')#reject
#plt.title('2019ICLR')
plt.ylabel('Number of papers',font1)
plt.xlabel('Date',font1)
#plt.tick_params(labelsize=13,rotation=12)
plt.tick_params(labelsize=15)
plt.legend(prop=font1)
plt.show()