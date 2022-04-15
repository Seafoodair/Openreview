from scipy.stats import pearsonr
import pandas as pd
a=[1,2,3]
b=[1,2,3]
df = pd.read_excel(r'C:/Users/wg/PycharmProjects/测试/enoslib-paper/ICLR2021-OpenReviewData/数据集处理/train.xlsx', 'Sheet2')
c=df['轮数']
d=df['字典长度']
print(pearsonr(c,d))