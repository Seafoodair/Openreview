#coding=utf-8
"""

date:Jan,20,2020
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
#When encountering difficulties, it is a good way to draw pictures by program.Wish All  A Happy Winter solstice
def opentxt(path):
    list=[]
    file=open(path,"r")
    for i in file.readlines():
        i=i.strip("\n")
        i=float(i)
        list.append(i)
    return list
if __name__ == '__main__':

    #filepath= 'accept2020.txt'
    #filepath2='accept2020avg.txt'
    #filepath3 = 'reject2020.txt'
    #filepath4 = 'reject2020avg.txt'
    #x = opentxt(filepath)
    #y = opentxt(filepath2)
    #j = opentxt(filepath3)
    #k = opentxt(filepath4)
    wr=pd.read_excel(os.path.abspath('../../data/cite and  score analysis data/2017try.xlsx'))
    wp=wr.cite
    #wp=sorted(wp)
    shoulu=wr.final_decision

    x=wr.id
    # plt.xticks([-5.0,-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0])
    # plt.yticks([0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0])

    # plt.xticks([-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    # plt.yticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # s是设置大小的。默认的maker为圆
    for j,k,z in zip(x,wp,shoulu):
        if z==0:
            p1 = plt.scatter(j, k, alpha=0.6, c='red', s=13)
        else:
            p1 = plt.scatter(j, k, alpha=0.6, c='green', s=13)
    """
    counter = 0
    for i, t in zip(x, y):
        for a, b in zip(j, k):
            if i == a and t == b:
                counter = counter + 1
        if counter == 0:
            p1 = plt.scatter(i, t, alpha=0.3, c='green', s=3, marker='o')
        else:
            p1 = plt.scatter(i, t, alpha=0.3, c='yellow', s=3, marker='o')
        counter = 0
    p2 = plt.scatter(j, k, alpha=0.4, c='red', label='reject', s=10)
    # for i, t in zip(j, k):
    #     p2 = plt.scatter(j, k, alpha=0.8, c='green', s=10)
    # set.majorGridlines = False
    """
    labels=['reject','accept']
    color=['red','green']
    alpha=[0.6,0.6]
    aa= [mpatches.Patch(color=color[i], label="{:s}".format(labels[i]), alpha=alpha[i]) for i in range(len(color))]
    #plt.title('2019acceptcite')
    plt.legend(handles=aa,loc='upper left',fontsize=15)
    # plt.xlabel('window')
    # plt.ylabel('std')

    plt.xlabel('submissions',fontsize=20)
    plt.ylabel('citation num',fontsize=20)
    #plt.tick_params(labelsize=15)
    plt.tick_params(labelsize=15)
    plt.grid(False)
    #plt.savefig('2017cite.pdf')
    plt.show()
