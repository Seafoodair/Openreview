#coding=utf-8
"""

date：Feb,17,2020

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
name_list = ['17\\01', '17\\02', '17\\03','17\\05','17\\06','17\\07','17\\08','17\\09','17\\10\n submission','17\\11','17\\12','18\\01\n notification','18\\02','18\\03','18\\04\n conference']
num_list = [1, 1, 3, 4, 5, 5,3,3,12,7,1,4,6,8,0]
num_list2 = [0, 0, 0, 6, 6, 2,3,1,6,6,1,2,5,2,0]
x = list(range(len(num_list)))
"""
total_width, n = 0.4, 2
width = total_width / n
plt.bar(x, num_list, width=width, label='accept', fc='green')#accept
for i in range(len(x)):
    x[i] += width
plt.bar(x, num_list2, width=width, label='reject', tick_label=name_list, fc='red')#reject
plt.title('2018ICLR')
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