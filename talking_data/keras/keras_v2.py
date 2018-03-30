
import os
import random
import warnings
import pickle

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['KERAS_BACKEND'] = 'tensorflow'

import pandas as pd
import numpy as np
import keras

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

SEED=42


np.random.seed(SEED)
random.seed(SEED)


TRAIN_ROWS = 184903890
VALID_ROWS = 53016937 # rows in train.csv with day == 2017-11-09             
TEST_ROWS_V0 = 57537505
TEST_ROWS = 18790469
CACHE = '../cache'

EMBEDS = [    
    {'name': 'app', 'dim': [768, 10]},
    {'name': 'channel', 'dim': [500, 5]},
    {'name': 'device', 'dim': [4227, 12]},
    {'name': 'ip', 'dim': [126420, 20]},
    {'name': 'os', 'dim': [956, 10]},
    {'name': 'hour', 'dim': [23, 5]}
]



#['app', 'channel', 'click_time', 'device', 'ip', 'is_attributed', 'os', 'hour', 'day']
class AnomalyDataset:
    def __init__(self, filter_ips=True):  
        with open(os.path.join(CACHE, 'train_test_base.pkl'), 'rb') as f:
            self.df = pickle.load(f)        
            self.df = self.df.reset_index(drop=True)
            assert len(self.df) == TRAIN_ROWS + TEST_ROWS_V0, \
                "%d %d" %(len(df), TRAIN_ROWS + TEST_ROWS_V0) 
        if filter_ips:
            self.df = self.df[self.df.ip < 126420]
        self.y = self.df.is_attributed.values               
        self.df.drop(['click_time', 'day', 'is_attributed'], axis=1, inplace=True)
        
    def build_input(self, X):
        embed_names = [e['name'] for e in EMBEDS]
        #misc_columns = [name for name in self.df.columns if name not in embed_names]
        #misc = X[misc_columns]
        #misc = None
        #h = {'misc': misc} 
        h = {}
        h.update(dict([(name, X.iloc[:, idx]) for idx, name in enumerate(embed_names)]))
        return h
            
    def parts(self, train_skip=80000000):
        skip = train_skip if train_skip else 0
        X_train = self.build_input(self.df.iloc[skip:TRAIN_ROWS-VALID_ROWS])
        y_train = self.y[skip:TRAIN_ROWS-VALID_ROWS]
        X_valid = self.build_input(self.df.iloc[TRAIN_ROWS-VALID_ROWS:TRAIN_ROWS])
        y_valid = self.y[TRAIN_ROWS-VALID_ROWS:TRAIN_ROWS]
        return X_train, X_valid, y_train, y_valid
    
    def train_test_ds(self):
        X = self.build_input(self.df)
        y = self.y
        return X, y




import keras
from keras.models import Model
from keras.layers import *
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.engine.topology import Layer
import keras.backend as K



def embed(input_dim, output_dim, x):
    e = Embedding(input_dim, output_dim, input_length=1, embeddings_regularizer=l2(1e-8))
    r = e(x)
    r = Reshape((output_dim,))(r)
    return r

def build_model():    
    #misc = Input(shape=(features,), name='misc')
    
    for e in EMBEDS:
        e['input'] = Input(shape=(1,), name=e['name'])
        e['layer'] = embed(e['dim'][0], e['dim'][1], e['input'])
        
    h = concatenate([e['layer'] for e in EMBEDS])
    h = BatchNormalization()(h)
    
    h = Dense(128, activation='relu')(h)
    h = BatchNormalization()(h)
    h = Dropout(0.5)(h)
    
    h = Dense(64, activation='relu')(h)
    #h = BatchNormalization()(h)
    h = Dropout(0.2)(h)
    
    h = Dense(32, activation='relu')(h)
    #h = BatchNormalization()(h)
    
    h = Dense(1, activation='sigmoid')(h)
    
    model = Model(inputs=[e['input'] for e in EMBEDS],
                  outputs=h)
    return model



import tensorflow as tf

def roc_auc_score_fn(y_true, y_pred):
    """ ROC AUC Score.
    
    Source:
    https://github.com/tflearn/tflearn/blob/1b0e4d3539e1c4c6b31735499328c27da3a874b0/tflearn/objectives.py
    
    Approximates the Area Under Curve score, using approximation based on
    the Wilcoxon-Mann-Whitney U statistic.
    Yan, L., Dodier, R., Mozer, M. C., & Wolniewicz, R. (2003).
    Optimizing Classifier Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic.
    Measures overall performance for a full range of threshold levels.
    `y_pred` and `y_true` must have the same type and shape.
    """
    pos = tf.boolean_mask(y_pred, tf.cast(y_true, tf.bool))
    neg = tf.boolean_mask(y_pred, ~tf.cast(y_true, tf.bool))

    pos = tf.expand_dims(pos, 0)
    neg = tf.expand_dims(neg, 1)

    # original paper suggests performance is robust to exact parameter choice
    gamma = 0.2
    p     = 3

    difference = tf.zeros_like(pos * neg) + pos - neg - gamma

    masked = tf.boolean_mask(difference, difference < 0.0)

    return tf.reduce_sum(tf.pow(-masked, p))


class CustomMetric(keras.callbacks.Callback):    
    def __init__(self, val_x, val_y):
        self.val_x = val_x
        self.val_y = val_y
        
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            preds = self.model.predict(self.val_x).ravel()
            logs['val-auc'] = roc_auc_score(self.val_y, preds)


class Classifier:
    def __init__(self, optimizer='adam', loss='binary_crossentropy'):
        # roc_auc_score_fn
        self.loss = loss
        self.optimizer = optimizer
           
    def train(self, ds, **fit_args):        
        train_input, dev_input, y_train, y_dev = ds.parts()
        
        model = build_model()
        model.compile(optimizer=self.optimizer, loss=self.loss)
        
        # callbacks        
        tensorboard = keras.callbacks.TensorBoard(log_dir='../tmp', write_graph=False)
        
        # val-auc is calculated by the custom eval_callback        
        ckpt = keras.callbacks.ModelCheckpoint(filepath='../tmp/weights.hdf5',
                                               save_weights_only=False,
                                               monitor='val-auc', mode='max',
                                               verbose=2, save_best_only=True)    
        eval_callback = CustomMetric(dev_input, y_dev)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val-auc', mode='max', 
                                                      patience=2)
        callbacks = [eval_callback, ckpt, reduce_lr, tensorboard]
        fit_args['callbacks'] = callbacks 
        fit_args['validation_data'] = (dev_input, y_dev)

        model.fit(train_input, y_train, **fit_args)

        model.load_weights('../tmp/weights.hdf5')

        print('best train-auc: {} val-auc: {}'.format(
            roc_auc_score(y_train, model.predict(train1_input).ravel()),
            roc_auc_score(y_dev, model.predict(dev_input).ravel())))
        
        return model
    
    def predict(self, model, X_test):
        print('.')



if __name__ == '__main__':
    print("loading")
    ds = AnomalyDataset()
    print("dataset loaded.")

    c = Classifier(optimizer='adam', loss=roc_auc_score_fn)
    model = c.train(ds, epochs=10, batch_size=64, shuffle=True, verbose=1)
