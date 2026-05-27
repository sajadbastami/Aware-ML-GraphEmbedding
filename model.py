import tensorflow as tf
from tensorflow.keras import layers, models
from gensim.models import Word2Vec
import numpy as np

def train_word2vec(sequences, vector_size=128, window=10):
    model = Word2Vec(sentences=sequences, vector_size=vector_size, window=window, min_count=1, workers=4)
    return model

def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        layers.GlobalMaxPooling1D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='sigmoid') # Sigmoid for Multi-label
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model