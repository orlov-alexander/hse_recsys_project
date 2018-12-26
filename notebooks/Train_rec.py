from tqdm import tqdm, trange
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, average_precision_score
import torch.nn as nn
from torch import optim
from itertools import product
import pickle
import numpy as np
import argparse
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import NLP_torch

activations = {'relu': nn.ReLU,
               'sigmoid': nn.Sigmoid,
               'tanh': nn.Tanh}
SEED = 100500
MIN_COUNT = 0
N_CORES = 10

parser = argparse.ArgumentParser()
parser.add_argument('--rootdir', type=str, help='Main path to logs (--logdir argument while launching Tensorboard)')
parser.add_argument('--logdir', type=str, help='logdir prefix for particular run')
parser.add_argument('--datafile', type=str, help='Path to pickled file with features and target')
parser.add_argument('--activation', type=str,
                    help='Activation function, one of the following: {}'.format(','.join(activations.keys())))
parser.add_argument('--n_epochs', type=int, help='Number of epochs')
# parser.add_argument('--metric', type=str, default='auc', help='metric_')
args = parser.parse_args()


with open(args.datafile, 'rb') as f:
    features, target = pickle.load(f)
y = np.vstack((target == 0, target == 1)).T.astype(np.int16)
# scaler = StandardScaler()
# features = scaler.fit_transform(features)
if MIN_COUNT > 0:
    cols_to_keep = y.sum(axis=0) > MIN_COUNT
    # Удалим также объекты, у которых после выпиливания редких тегов не остается меток.
    rows_to_keep = y[:, cols_to_keep].sum(axis=1) > 0
    features = features[rows_to_keep, :]
    y = y[np.ix_(rows_to_keep, cols_to_keep)]

kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
train_idx, test_idx = next(kf.split(y))
# X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.33, random_state=SEED)

# Сетка параметров
hidden_sizes = [3000, 100, 200]
optimizers = [optim.RMSprop, optim.Adam]
lr = [1e-2]
bs = [30000, 60000, 70000]
dropout_rates = [0.05, 0.1, 0.2, 0.3]
params_sets = product(hidden_sizes, optimizers, lr, bs, dropout_rates)
n_epochs = args.n_epochs

# Собственно обучение
for hidden_size, opt, learning_rate, batch_size, dropout_rate in tqdm(list(params_sets)):
    model = NLP_torch.Perceptron(num_features=features.shape[1],
                                 n_classes=y.shape[1],
                                 hidden_size=hidden_size,
                                 activation=activations[args.activation],
                                 dropout_rate=dropout_rate)
    logdir = f'{args.rootdir}/{args.logdir}_hidden_size={hidden_size}_opt={opt}' + \
             f'_learning_rate={learning_rate}_batch_size={batch_size}_dropout_rate={dropout_rate}_nepochs={n_epochs}'
    trainer = NLP_torch.ModelTraining(model=model,
                                      optimizer=opt,
                                      optimizer_params={'lr': learning_rate},
                                      loss_fun=nn.BCELoss(),
                                      batch_size=batch_size,
                                      log_dir=logdir,
                                      scheduler=optim.lr_scheduler.ReduceLROnPlateau,
                                      scheduler_params={'threshold': 1e-4, 'patience': 1})
    # trainer.writer.add_text('metrics/mean_roc_text', bytes(str(np.sum(np.array([1., 2.]))), encoding='utf-8'), 0)
    train_log, val_log = trainer.train(n_epochs, features[train_idx, :], y[train_idx],
                                       features[test_idx, :], y[test_idx])
    t_start = datetime.now()
    y_pred = trainer.predict_proba(features[test_idx, :])
    elapsed = datetime.now() - t_start
    print('Time to infer on test set: {}'.format(elapsed))

    print(f'hidden_size={hidden_size}_opt={opt}_learning_rate={learning_rate}' +
          f'_batch_size={batch_size}_dropout_rate={dropout_rate})')
    # print('#' * 20)
    # trainer.writer.add_histogram('metrics/prediction_distribution', y_pred.reshape(-1), 0, bins=20)
    roc = roc_auc_score(target[test_idx], y_pred[:, 1])
    prc = average_precision_score(target[test_idx], y_pred[:, 1])
    trainer.writer.add_scalar('metrics/mean_roc', roc, 0)
    trainer.writer.add_scalar('metrics/average_precision', prc, 0)

    # trainer.writer.add_text('metrics/mean_roc_text', bytes(str(np.mean(roc)), encoding='utf-8'), 0)
    # trainer.writer.add_text('metrics/mean_prc_text', bytes(str(np.mean(prc)), encoding='utf-8'), 0)
