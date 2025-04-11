
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam

def build_hybrid_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    
    encoded = Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(0.0001))(input_layer)
    encoded = LeakyReLU(negative_slope=0.1)(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = Dropout(0.5)(encoded)  
    
    encoded = Dense(512)(encoded)
    encoded = LeakyReLU(negative_slope=0.1)(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = Dropout(0.3)(encoded)
    
    encoded = Dense(256)(encoded)
    encoded = LeakyReLU(negative_slope=0.1)(encoded)
    
    decoded = Dense(512)(encoded)
    decoded = LeakyReLU(negative_slope=0.1)(decoded)
    decoded = Dense(1024)(decoded)
    decoded = LeakyReLU(negative_slope=0.1)(decoded)
    output_layer = Dense(input_dim, activation='relu')(decoded)
    
    anomaly_score = Dense(1, activation='sigmoid', name="anomaly_score")(encoded)
    
    autoencoder = Model(input_layer, [output_layer, anomaly_score])
    
    def hybrid_loss(y_true, y_pred):
        reconstruction_loss = tf.reduce_mean(tf.square(y_true[0] - y_pred[0]))
        classification_loss = tf.keras.losses.BinaryCrossentropy()(y_true[1], y_pred[1])
        return reconstruction_loss + 0.03 * classification_loss 
    
    autoencoder.compile(optimizer=Adam(learning_rate=0.00005), loss=hybrid_loss)
    return autoencoder

# Exemplo de uso:
# hybrid_autoencoder = build_hybrid_autoencoder(X_train_scaled.shape[1])
# y_train = [X_train_scaled, y_train_labels.reshape(-1, 1)]
# hybrid_autoencoder.fit(X_train_scaled, y_train, epochs=150, batch_size=128, verbose=1, validation_split=0.2)
# X_test_reconstructed, X_test_anomaly = hybrid_autoencoder.predict(X_test_scaled)
# test_reconstruction_errors = np.mean(np.abs(X_test_scaled - X_test_reconstructed), axis=1)
# threshold = np.percentile(test_reconstruction_errors, 65)
# y_pred = (test_reconstruction_errors > threshold).astype(int)
# y_pred = np.round((0.5 * y_pred + 0.5 * X_test_anomaly.flatten()))
