import numpy as np
import pandas as pd
from bert4keras.backend import keras, set_gelu
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.tokenizers import Tokenizer
from keras.layers import Lambda, Dense
import os
set_gelu('tanh')
config_path = os.path.abspath('../../data/sentiment analysis data/')+'/electra_small/bert_config_tiny.json'
checkpoint_path = os.path.abspath('../../data/sentiment analysis data/')+'/electra_small/electra_small'
dict_path = os.path.abspath('../../data/sentiment analysis data/')+'/electra_small/vocab.txt'
tokenizer = Tokenizer(dict_path, do_lower_case=True)
num_classes = 3#class
max_len = 128#
batch_size = 32#
def read_message(path):
    data = pd.read_csv(path,delimiter='\t').values.tolist()
    random_order = list(range(len(data)))
    np.random.shuffle(random_order)
    train_data = [data[j] for i, j in enumerate(random_order) if i % 10 != 0 and i % 10 != 1]
    valid_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 0]
    test_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 1]
    return train_data, valid_data, test_data
    # train model


# load dataset
train_data, valid_data, test_data = read_message(os.path.abspath('../../data/sentiment analysis data/sentimentanalysistraindata/novel.csv'))



class data_generator(DataGenerator):
    """date-generator
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, max_length=max_len)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# load pre-model
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='electra',
    return_keras_model=False,

)  # bulid model and load weights

output = Lambda(lambda x: x[:, 0],
                name='CLS-token')(bert.model.output)
output = Dense(units=num_classes,
               activation='softmax',
               kernel_initializer=bert.initializer)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()
AdamLR = extend_with_piecewise_linear_lr(Adam)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(2e-3),  # 用足够小的学习率
    # optimizer=AdamLR(learning_rate=1e-3,
    #                  lr_schedule={1000: 1, 2000: 0.1}),
    metrics=['accuracy'],
)

train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('best_model.weights')
        test_acc = evaluate(test_generator)
        print(u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
              (val_acc, self.best_val_acc, test_acc))


evaluator = Evaluator()
model.fit_generator(train_generator.forfit(),
                    steps_per_epoch=len(train_generator),
                    epochs=10,
                    callbacks=[evaluator])

model.load_weights('best_model.weights')
print(u'final test acc: %05f\n' % (evaluate(test_generator)))
