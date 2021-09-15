from data_loader import CNN_train_data_loader, CNN_test_data_loader, LSTM_train_data_loader, LSTM_test_data_loader
from model import model
from hyperparams import Hyperparams
from paths import Paths
import tensorflow as tf
from logger import NBatchLogger
from tensorflow.keras.callbacks import TensorBoard
import time
from tensorflow.keras.models import save_model, load_model
import logging
from tensorflow.keras.losses import BinaryCrossentropy
import os
from timer import timecallback
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


logger configuration
FORMAT = "[%(filename)s: %(lineno)3s] %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

name = 'CNN-Autoencoder-{}'.format(int(time.time()))

H = Hyperparams()
P = Paths()
nbl = NBatchLogger(display=1)
m = model()

tensorboard = TensorBoard(log_dir=os.path.join(P.log_dir, name))

adam = tf.keras.optimizers.Adam(lr=H.learning_rate)
rmsprop = tf.keras.optimizers.RMSprop(lr=H.learning_rate)
if os.path.exists(P.cnn_encoder):
    cnn_encoder = load_model(P.cnn_encoder)

else:
    cnn_train_data_loader = CNN_test_data_loader(H.train_batch_size)
    logger.info("cnn_train_data_loader created")
    cnn_encoder, cnn_autoencoder = m.get_model_cnn()
    logging.info("cnn_autoencoder created")
    cnn_autoencoder.compile(optimizer=adam, loss='mean_squared_error')
    logging.info("cnn_model compiled, starting training ...")
    cnn_autoencoder.fit(x=cnn_train_data_loader, epochs=H.cnn_num_epochs, verbose=2, workers=6, use_multiprocessing=True, max_queue_size=100, 
        shuffle=False, callbacks=[nbl, tensorboard])
    save_model(cnn_encoder, P.cnn_encoder)
    logging.info("saved cnn_model")

name = 'LSTM-{}'.format(int(time.time()))
tensorboard_LSMT = TensorBoard(log_dir=os.path.join(P.log_dir, name))
timePerEpoch = timecallback()

nbl_lstm = NBatchLogger(display=1)
lstm_test_data_loader = LSTM_test_data_loader(H.test_batch_size)
lstm_train_data_loader = LSTM_train_data_loader(H.train_batch_size)


if(os.path.exists(P.lstm)):
    lstm_model = load_model(P.lstm)
else:
    print("lstm_train_data_loader initialized")
    lstm_model = m.get_model_lstm()
    print("lstm_model loaded")
    lstm_model.compile(optimizer=rmsprop, loss=BinaryCrossentropy(), metrics=['accuracy'])
    print("lstm compiled, starting training ...")
    lstm_model.fit(x=lstm_train_data_loader, epochs=H.lstm_num_epochs, verbose=2, shuffle=False,
    workers=6, use_multiprocessing=True, max_queue_size=200,callbacks=[nbl_lstm, tensorboard_LSMT, timePerEpoch])
    save_model(lstm_model, P.lstm)


print("*"*5," Training Metrics ", "*"*5)

train_labels = []
train_preds = []
for train_batch, train_label_batch in lstm_train_data_loader:
    train_labels.extend(list(train_label_batch))
    for k in list(lstm_model.predict_on_batch(train_batch)):
        if(k[0].numpy() >= 0.3):
            train_preds.append(1)
        else:
            train_preds.append(0)



test_labels = []
test_preds = []
for test_batch, label_batch in lstm_test_data_loader:
    test_labels.extend(list(label_batch))
    for k in list(lstm_model.predict_on_batch(test_batch)):
        if(k[0].numpy() >= H.threshold):
            test_preds.append(1)
        else:
            test_preds.append(0)
    #test_preds.extend(list(lstm_model.predict_on_batch(test_batch)))
    #print(test_labels, test_preds[0])

  