"""
"""

import os
import random
import warnings

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


class CachedDataset:
    def __init__(self):
        self.train_csv = '../cache/lgbm_train.csv'
        self.valid_csv = '../cache/lgbm_valid.csv'
        with open('../cache/lgbm_features.txt', 'r') as f:
            self.feature_names = f.readlines()[0].strip().split(',')
            self.feature_names = self.feature_names[1:]
        
    def parse_csv(self, fname):
        X = pd.read_csv(fname, nrows=50000000)#TODO
        X.columns = ['is_attributed'] + self.feature_names
        y = X[['is_attributed']].values
        X.drop(['is_attributed'], axis=1, inplace=True)        
        return X, y        
        
    def build(self):
        self.X_train, self.y_train = self.parse_csv(self.train_csv)
        self.X_valid, self.y_valid = self.parse_csv(self.valid_csv)
        self.X_train = self.build_input(self.X_train)
        self.X_valid = self.build_input(self.X_valid)
            
    def build_input(self, X):
        embed_names = [e['name'] for e in EMBEDS]
        # misc = np.concatenate([self.X[:, 0], self.X[:, idx], ...]) #TODO   
        #misc_names = [name for name in self.feature_names if name not in embed_names]
        #print(misc_names)
        #misc = np.hstack([X.iloc[:, idx] for idx, name in enumerate(self.feature_names) if name not in embed_names])
        #print(misc.shape)
        misc = X.iloc[:, 5:]
        h = {'misc': misc} 
        h.update(dict([(name, X.iloc[:, idx]) for idx, name in enumerate(embed_names)]))
        return h
            
    def parts(self):
        return self.X_train, self.X_valid, self.y_train, self.y_valid
    
import keras
from keras.models import Model
from keras.layers import *
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.engine.topology import Layer
import keras.backend as K

D = 14

# Important to keep same ordering as in train.npz & test.npz            
EMBEDS = [    
    {'name': 'app', 'dim': [2**D, 100]},
    {'name': 'device', 'dim': [2**D, 50]},
    {'name': 'os', 'dim': [2**D, 50]},
    {'name': 'channel', 'dim': [2**D, 50]},
    {'name': 'hour', 'dim': [24, 5]},    
]

def embed(input_dim, output_dim, x):
    e = Embedding(input_dim, output_dim, input_length=1, embeddings_regularizer=l2(1e-8))
    r = e(x)
    r = Reshape((output_dim,))(r)
    return r

def build_model(features):    
    misc = Input(shape=(features,), name='misc')
    
    for e in EMBEDS:
        e['input'] = Input(shape=(1,), name=e['name'])
        e['layer'] = embed(e['dim'][0], e['dim'][1], e['input'])
        
    h = concatenate([misc] + [e['layer'] for e in EMBEDS])
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
    
    model = Model(inputs=[misc] + [e['input'] for e in EMBEDS],
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
            logs['val-auc'] = roc_auc_score(self.val_y, self.model.predict(self.val_x).ravel())
            
class Classifier:
    def __init__(self, optimizer='adam', loss='binary_crossentropy'):
        # roc_auc_score_fn
        self.loss = loss
        self.optimizer = optimizer
        
    def resume(self, model, ds, **fit_args):
        train1_input, dev_input, y_train, y_dev = ds.parts()
        use_validation = dev_input is not None

        if len(train1_input['misc'].shape) == 1:
            features = 1
        else:
            features = train1_input['misc'].shape[1]
                    
        # callbacks        
        tensorboard = keras.callbacks.TensorBoard(log_dir='../tmp', write_graph=False)
        if use_validation:
            ckpt = keras.callbacks.ModelCheckpoint(filepath='../tmp/weights.hdf5', 
                                                   # val-auc is calculated by the custom eval_callback
                                                   monitor='val-auc', mode='max',
                                                   verbose=2, save_best_only=True)    
            eval_callback = CustomMetric(dev_input, y_dev)
            reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val-auc', mode='max', 
                                                          patience=2)
            callbacks = [eval_callback, ckpt, reduce_lr, tensorboard]
            fit_args['callbacks'] = callbacks 
            fit_args['validation_data'] = (dev_input, y_dev)
        else:
            ckpt = keras.callbacks.ModelCheckpoint(filepath='../tmp/weights.hdf5', 
                                                   verbose=2, save_best_only=False)
            callbacks = [ckpt]
            fit_args['callbacks'] = callbacks 
            fit_args['validation_data'] = None
            fit_args['validation_split'] = 0.05

        model.fit(train1_input, y_train, **fit_args)

        model.load_weights('../tmp/weights.hdf5')

        if use_validation:
            print('fold {}, best train-auc: {} val-auc: {}'.format(
                fold_no, 
                None, #roc_auc_score(y_train, model.predict(train1_input).ravel()),
                roc_auc_score(y_dev, model.predict(dev_input).ravel())))
        
        return model

    
    def train(self, fold_no, ds, **fit_args):
        
        train1_input, dev_input, y_train, y_dev = ds.parts()
        use_validation = dev_input is not None

        if len(train1_input['misc'].shape) == 1:
            features = 1
        else:
            features = train1_input['misc'].shape[1]
            
        model = build_model(features)
        model.compile(optimizer=self.optimizer, loss=self.loss)
        
        # callbacks        
        tensorboard = keras.callbacks.TensorBoard(log_dir='../tmp', write_graph=False)
        if use_validation:
            ckpt = keras.callbacks.ModelCheckpoint(filepath='../tmp/weights.hdf5', 
                                                   # val-auc is calculated by the custom eval_callback
                                                   monitor='val-auc', mode='max',
                                                   verbose=2, save_best_only=True)    
            eval_callback = CustomMetric(dev_input, y_dev)
            reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val-auc', mode='max', 
                                                          patience=2)
            callbacks = [eval_callback, ckpt, reduce_lr, tensorboard]
            fit_args['callbacks'] = callbacks 
            fit_args['validation_data'] = (dev_input, y_dev)
        else:
            ckpt = keras.callbacks.ModelCheckpoint(filepath='../tmp/weights.hdf5', 
                                                   verbose=2, save_best_only=False)
            callbacks = [ckpt]
            fit_args['callbacks'] = callbacks 
            fit_args['validation_data'] = None
            fit_args['validation_split'] = 0.05

        model.fit(train1_input, y_train, **fit_args)

        model.load_weights('../tmp/weights.hdf5')

        if use_validation:
            print('fold {}, best train-auc: {} val-auc: {}'.format(
                fold_no, 
                roc_auc_score(y_train, model.predict(train1_input).ravel()),
                roc_auc_score(y_dev, model.predict(dev_input).ravel())))
        
        return model
    
    def predict(self, model, X_test):
        print('.')
        
ds = CachedDataset()
ds.build()

c = Classifier(optimizer='sgd', loss=roc_auc_score_fn)
model = c.train(0, ds, epochs=20, batch_size=64, shuffle=True)
