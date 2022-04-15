from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, Flatten
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Convolution1D
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers
import pandas as pd
import re
import nltk  # 安装导入词向量计算工具
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model  # 加载模型
import matplotlib.pyplot as plt
from tensorflow.python.ops.math_ops import reduce_prod
def clean_text(text):
    # 用正则表达式取出符合规范的部分
    text = re.sub(r'[^ws]', '', text, re.UNICODE)
    ##小写化所有的词，并转成词list
    text = text.lower()
    ##第一个参数表示待处理单词，必须是小写的；第二个参数表示POS，默认为NOUN
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text
# nltk.download('omw-1.4')
df= pd.read_excel(r'C:/Users/wg/PycharmProjects/测试/enoslib-paper/ICLR2021-OpenReviewData/数据集处理/train.xlsx','Sheet2')#读数据集
#df1 = df1.drop(['id'], axis=1)
# print(df1.head())
"""
df2 = pd.read_csv('C:/Users/wg/PycharmProjects/测试/enoslib-paper/ICLR2021-OpenReviewData/data/imdb_master.csv',
                  encoding="latin-1")
# print(df2.head())
df2 = df2.drop(['Unnamed: 0', 'type', 'file'], axis=1)
df2.columns = ["review", "sentiment"]
# print(df2.head())
# 对数据进行处理。
df2 = df2[df2.sentiment != 'unsup']
df2['sentiment'] = df2['sentiment'].map({'pos': 1, 'neg': 0})
# print(df2.head())
# 合并数据 
df = pd.concat([df1, df2]).reset_index(drop=True)
"""
# print(df.head())
# print(df.info())
"""
MEDIUM_SIZE=12
#数据可视化sentiment比例分布
plt.hist(df[df.sentiment == 1].sentiment,
         bins=2, color='green', label='Positive')
plt.hist(df[df.sentiment == 0].sentiment,
         bins=2, color='blue', label='Negative')
plt.title('Classes distribution in the train data', fontsize=MEDIUM_SIZE)
plt.xticks([])
plt.xlim(-0.5, 2)
plt.legend()
plt.show()
"""
stop_words = set(nltk.corpus.stopwords.words('english'))  # 停词
lemmatizer = WordNetLemmatizer()  # 提取单词的主干
#数据集长度
l=len(df['review'])
df['Processed_Reviews'] = df['review'].apply(lambda x: clean_text(x))
df['rebuttal_deal']=df['reb'].apply(lambda x: clean_text(x))
print(df.head())
print("长度",len(df['reb']))
df.Processed_Reviews.apply(lambda x: len(x.split(" "))).mean()
df.rebuttal_deal.apply(lambda x: len(x.split(" "))).mean()
'''
def lemmatize(tokens: list) -> list:
    # 1. Lemmatize 词形还原 去掉单词的词缀 比如，单词“cars”词形还原后的单词为“car”，单词“ate”词形还原后的单词为“eat”
    tokens = list(map(lemmatizer.lemmatize, tokens))
    lemmatized_tokens = list(map(lambda x: lemmatizer.lemmatize(x, "v"), tokens))
    # 2. Remove stop words 删除停用词
    meaningful_words = list(filter(lambda x: not x in stop_words, lemmatized_tokens))
    return meaningful_words
def preprocess(review: str, total: int, show_progress: bool = True) -> list:
    if show_progress:
        global counter
        counter += 1
        print('Processing... %6i/%6i'% (counter, total), end='r')
    # 1. Clean text
    review = clean_review(review)
    # 2. Split into individual words
    tokens = word_tokenize(review)
    # 3. Lemmatize
    lemmas = lemmatize(tokens)
    # 4. Join the words back into one string separated by space,
    # and return the result.
    return lemmas
'''
max_features = 6000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(df['Processed_Reviews'])
tokenizer.fit_on_texts(df['rebuttal_deal'])
list_tokenized_train = tokenizer.texts_to_sequences(df['Processed_Reviews'])
list_tokenized_train2 = tokenizer.texts_to_sequences(df['rebuttal_deal'])
maxlen = 130
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_r = pad_sequences(list_tokenized_train2, maxlen=maxlen)
yr= df['ca']
yrr=yr
y = df['label']#这里是映射 label 到sentiment
print(y)
y= to_categorical(y, num_classes=4)#[1,3,6,8]
yr= to_categorical(yr, num_classes=5)#[0,1,2,3,4]

embed_size = 128
model = Sequential()
model.add(Embedding(max_features, embed_size))  # (1,130,128)
model.add(Bidirectional(LSTM(64, return_sequences=True)))  # 输出（1,130,128=64*2）
model.add(Bidirectional(LSTM(32, return_sequences=True)))
model.add(GlobalMaxPool1D())
model.add(Dense(20, activation="relu"))
model.add(Dropout(0.05))
model.add(Dense(4, activation="sigmoid"))#这里分为4类 review 分类
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
batch_size = 1000
epochs = 10
model.fit(X_t[:l-500], y[:l-500], batch_size=batch_size, epochs=epochs, validation_split=0.1)#暂时先别训练
# 模型保存
model.save('lstm_modelzuizuizu.h5')

"""
生成提交结果。
y_pred = model.predict(X_te)
def submit(predictions):
    df_test['sentiment'] = predictions
    df_test.to_csv('submission.csv', index=False, columns=['id','sentiment'])

submit(y_pred)
"""
layer_model = Model(inputs=model.input, outputs=model.layers[1].output)
# 读取到Bilstm层
import sys
import numpy as np
listanswer=[]
for i in X_r:
    i = i[np.newaxis, :]
    #print(i)
    #print(type(i))  # <class 'numpy.ndarray'>
    # print()
    # layer_model2.add_metric() 可能这个就是那个函数
    #em = layer_model.predict(i)
    #print(em.shape)
    #print(em)
    shuchu = layer_model.predict(i)#bilstm 输出
    #print('------------------------')
    #print(shuchu)
    #print(type(shuchu))
    #print(shuchu.shape)
    listanswer.append(shuchu[0][129])
    #combine = tf.add(em, shuchu)
    #print("@@@@@@@@@@@@@@@@@@@")
    #print(combine)
    #print(combine.shape)  # 这个是组合的东西。拼接成功。
    # ccc=shuchu[129][0]
    #
    # print(ccc)
    # print(len(ccc))
    # print(ccc.shape)
    #sys.exit(0)
    # tf.stack 这个函数是张量堆叠。
print("成功了一半")
#listanswer转换为np.array
wg=np.array(listanswer)
#wge = np.zeros((1000,2))#这个改为1166
print('尺寸',wg.shape)
wge = np.zeros((l,2)) #数据集长度
wg=np.hstack((wg,wge))
print("这个shape是：")
print(wg.shape)

#这个是第二个model
#print(X_t[0:1000].shape)
print("第一个结果")
#X_t[0:1000]=np.expand_dims(X_t[0:1000], 1)
print(X_t[0:1000].shape)
class MyModel(tf.keras.Model):

  def __init__(self):
    super().__init__()
    #self.sq = tf.keras.Sequential()

    self.emd = tf.keras.layers.Embedding(6000, 128)
    self.dense1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))
    self.pool = tf.keras.layers.GlobalMaxPool1D()
    self.dense = tf.keras.layers.Dense(20, activation="relu")
    self.dropout = tf.keras.layers.Dropout(0.05)
    self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.sigmoid)

  def call(self, inputs):
    #print(inputs.shape)
    #x = self.sq()
    inputs1 = inputs[:,0:130]
    print(inputs1.shape,type(inputs1))
    inputs2 = inputs[:,130:258]
    print(inputs2.shape,type(inputs2))
    x1=self.emd(inputs1)
    #xemd=self.emd(inputs2)
    #fin=inputs2[0][129]
    #print(fin.shape)
    #print(xemd.shape)
    x3=self.dense1(x1)
    print(x3.shape)
    #如果把第一行代码当作单词拼上。
    for i in range(130):
    #inputs2=np.expand_dims(inputs2,0).repeat(130,axis=0)#扩展维度
        input22=tf.expand_dims(inputs2,1)
    print("inputs2的结果：")
    print(input22.shape)
    x3=tf.add(x3,input22)
    #print(x3.shape)
    x4=self.pool(x3)
    x5=self.dense(x4)
    x6=self.dropout(x5)
    return self.dense2(x6)


import numpy as np
model2 = MyModel()
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
batch_size = 1000
epochs2 = 10
model2.fit(np.hstack([X_t[:l-500],wg[:l-500]]), yr[:l-500], batch_size=batch_size, epochs=epochs2, validation_split=0.2)
#input_test=np.hstack([X_t[l-500:l],wg[l-500:l])
input_test=np.hstack([X_t[l-500:],wg[l-500:]])#这个输入可能还有问题
pred_test = np.argmax(model2.predict(input_test), axis=1)

map_label={0: '0', 1: '2', 2: '3', 3: '5', 4: '7'}
yt=yrr[l-500:]
print(type(yt))#输出yt的类型
print("-----")
print(yt.shape)
print(pred_test.shape)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
#yt=np.argmax(yt)  #这个是降维
print(classification_report([map_label[i] for i in yt], [map_label[i] for i in pred_test]))
print("老子搞定了")
