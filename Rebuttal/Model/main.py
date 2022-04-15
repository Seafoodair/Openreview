import os
import random
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm#这里删掉了notebook
import matplotlib.pyplot as plt
from attention import Attention
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K
from transformers import AutoTokenizer
from transformers import BertConfig
from transformers import TFBertModel
from transformers import TFAutoModel, AutoTokenizer, BertConfig, TFBertModel
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=25)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=15)
    plt.yticks(tick_marks, classes, fontsize=15)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize = 14)

    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)
### READ DATA ###
df = pd.read_excel(r'./train.xlsx', 'Sheet2')
#df = pd.read_json('./News_Category_Dataset_v2.json', lines=True) #读json 文件 按行读取
#df.category.value_counts 对类别计数
#df = df[df.category.isin(df.category.value_counts().tail(8).index)].copy() # select 8 low dimensional categories
#map_label = dict(enumerate(df.category.factorize()[1]))
map_label={0: 'LATINO VOICES', 1: 'EDUCATION', 2: 'COLLEGE', 3: 'ARTS & CULTURE', 4: 'GOOD NEWS'}
#df['category'] = df.category.factorize()[0]

print(df.shape)
print(df.head())


### UTILITY FUNCTIONS FOR TOKENIZATIONS, MASKS AND SEGMENTS CREATION ###
### from: https://www.kaggle.com/akensert/bert-base-tf2-0-now-huggingface-transformer

def set_seed(seed):
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


def convert_to_transformer_inputs(str1, str2, tokenizer, max_sequence_length, double=True):
    def return_id(str1, str2, truncation_strategy, length):

        inputs = tokenizer.encode_plus(str1, str2,
                                       add_special_tokens=True,
                                       max_length=length,
                                       truncation_strategy=truncation_strategy)

        input_ids = inputs["input_ids"]
        input_masks = [1] * len(input_ids)
        input_segments = inputs["token_type_ids"]

        padding_length = length - len(input_ids)
        padding_id = tokenizer.pad_token_id

        input_ids = input_ids + ([padding_id] * padding_length)
        input_masks = input_masks + ([0] * padding_length)
        input_segments = input_segments + ([0] * padding_length)

        return [input_ids, input_masks, input_segments]

    if double:

        input_ids_1, input_masks_1, input_segments_1 = return_id(
            str1, None, 'longest_first', max_sequence_length)

        input_ids_2, input_masks_2, input_segments_2 = return_id(
            str2, None, 'longest_first', max_sequence_length)

        return [input_ids_1, input_masks_1, input_segments_1,
                input_ids_2, input_masks_2, input_segments_2]

    else:

        input_ids, input_masks, input_segments = return_id(
            str1, str2, 'longest_first', max_sequence_length)

        return [input_ids, input_masks, input_segments,
                None, None, None]


def compute_input_arrays(df, columns, tokenizer, max_sequence_length, double=True):
    input_ids_1, input_masks_1, input_segments_1 = [], [], []
    input_ids_2, input_masks_2, input_segments_2 = [], [], []

    # print('1111111111111111111111111111111111')
    # print(df[columns].info())
    # print(type(df[columns]))
    # print(df[columns].iterrows())
    # print(type(df[columns].iterrows()))
    # print('22222222222222')

    for _, instance in tqdm(df[columns].iterrows(), total=len(df)):


        str1, str2 = instance[columns[0]], instance[columns[1]]

        ids_1, masks_1, segments_1, ids_2, masks_2, segments_2 = \
            convert_to_transformer_inputs(str1, str2, tokenizer, max_sequence_length, double=double)

        input_ids_1.append(ids_1)
        input_masks_1.append(masks_1)
        input_segments_1.append(segments_1)

        input_ids_2.append(ids_2)
        input_masks_2.append(masks_2)
        input_segments_2.append(segments_2)

    if double:

        return [np.asarray(input_ids_1, dtype=np.int32),
                np.asarray(input_masks_1, dtype=np.int32),
                np.asarray(input_segments_1, dtype=np.int32),
                np.asarray(input_ids_2, dtype=np.int32),
                np.asarray(input_masks_2, dtype=np.int32),
                np.asarray(input_segments_2, dtype=np.int32)]

    else:

        return [np.asarray(input_ids_1, dtype=np.int32),
                np.asarray(input_masks_1, dtype=np.int32),
                np.asarray(input_segments_1, dtype=np.int32)]
### TRAIN TEST SPLIT ###

#X_train, X_test, y_train, y_test = train_test_split(df[['headline','short_description']], df['category'].values,
#                                                random_state=33, test_size = 0.3)

X_train, X_test, y_train, y_test = train_test_split(df[['review','reb']], df['ca'].values,
                                                    random_state=33, test_size = 0.1)
#print(X_train)
#print(df['category'].values)
del df

#print(X_train.shape, X_test.shape)
#print(y_train.shape, y_test.shape)
### IMPORT TOKENIZER ###

MAX_SEQUENCE_LENGTH = 300

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
### CREATE SEQUENCES (id, mask, segments) FOR TRAIN AND TEST ###

#input_train = compute_input_arrays(X_train, ['headline','short_description'], tokenizer, MAX_SEQUENCE_LENGTH)
#input_test = compute_input_arrays(X_test, ['headline','short_description'], tokenizer, MAX_SEQUENCE_LENGTH)
input_train = compute_input_arrays(X_train, ['review','reb'], tokenizer, MAX_SEQUENCE_LENGTH)#df[[]]
input_test = compute_input_arrays(X_test, ['review','reb'], tokenizer, MAX_SEQUENCE_LENGTH)


def dual_bert():
    set_seed(33)

    opt = Adam(learning_rate=2e-5)

    id1 = Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    id2 = Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)

    mask1 = Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    mask2 = Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)

    atn1 = Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    atn2 = Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)

    config = BertConfig()
    config.output_hidden_states = False  # Set to True to obtain hidden states
    bert_model1 = TFBertModel.from_pretrained('bert-base-uncased', config=config)
    bert_model2 = TFBertModel.from_pretrained('bert-base-uncased', config=config)

    embedding1 = bert_model1(id1, attention_mask=mask1, token_type_ids=atn1)[0]
    embedding2 = bert_model2(id2, attention_mask=mask2, token_type_ids=atn2)[0]
    x=Concatenate()([embedding1, embedding2])
    x = keras.layers.Bidirectional(  # 加上这个就变成了双向lstm
        keras.layers.LSTM(  # 这个是单向lstm
            64,
            # 权重初始化
            kernel_initializer='he_normal',
            # 返回每个token的输出，如果设置为False 只出最后一个。
            return_sequences=True
        ))(x)
    #x = Lambda(lambda x: x[:, 0], name='CLS-token')(x)#降维
    #x1 = GlobalAveragePooling1D()(embedding1)
    #x2 = GlobalAveragePooling1D()(embedding2)

    #x = Concatenate()([x1, x2])
    x = Attention(128)(x)  # 加入attention

    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    #out = Dense(len(map_label), activation='softmax')(x)
    out = Dense(5, activation='softmax')(x)

    model = Model(inputs=[id1, mask1, atn1, id2, mask2, atn2], outputs=out)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt,metrics=['accuracy'])#加个评测指标

    return model
model = dual_bert()
model.fit(input_train, y_train, epochs=10, batch_size=6 )
#打印网络结构
model.summary()
### PREDICT TEST ###

pred_test = np.argmax(model.predict(input_test), axis=1)
print(classification_report([map_label[i] for i in y_test], [map_label[i] for i in pred_test]))
cnf_matrix = confusion_matrix([map_label[i] for i in y_test],
                              [map_label[i] for i in pred_test])

plt.figure(figsize=(7,7))
plot_confusion_matrix(cnf_matrix, classes=list(map_label.values()))
plt.show()