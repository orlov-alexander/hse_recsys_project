import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import numpy as np
from scipy import sparse
import subprocess
import re
from typing import List, Callable, Union


class LogReg(nn.Module):
    def __init__(self, num_features: int, n_classes: int, hidden_size: int, activation: nn.Module, dropout_rate: float):
        super(LogReg, self).__init__()
        self.n_classes = n_classes
        self.network = nn.Sequential(nn.Linear(num_features, hidden_size),
                                     nn.BatchNorm1d(hidden_size),
                                     nn.Dropout(dropout_rate),
                                     activation(),
                                     nn.Linear(hidden_size, n_classes))

    def forward(self, inp):
        out = self.network(inp)
        probas = F.softmax(out, dim=-1)
        return probas


class Perceptron(nn.Module):
    def __init__(self, num_features: int, n_classes: int, hidden_size: int, activation: nn.Module, dropout_rate: float):
        super(Perceptron, self).__init__()
        '''
        self.dtype = dtype
        dims = [num_features] + list(hidden_size) + [n_classes]
        fc_layer_sizes = [(d1, d2) for d1, d2 in zip(dims[:-1], dims[1:])]
        fc_layers = [nn.Linear(in_features=d1, out_features=d2) for d1, d2 in fc_layer_sizes]
        '''
        self.n_classes = n_classes
        self.network = nn.Sequential(nn.Linear(num_features, hidden_size),
                                     nn.BatchNorm1d(hidden_size),
                                     nn.Dropout(dropout_rate),
                                     activation(),
                                     nn.Linear(hidden_size, hidden_size),
                                     nn.BatchNorm1d(hidden_size),
                                     nn.Dropout(dropout_rate),
                                     activation(),
                                     nn.Linear(hidden_size, n_classes))

    def forward(self, inp):
        out = self.network(inp)
        probas = F.softmax(out, dim=-1)
        return probas


# TODO: опционально добавить дополнительные признаки (процент капслока, длина в словах)
class Conv1dText(nn.Module):
    def __init__(self, n_features: int, n_classes: int, max_length: int):
        super(Conv1dText, self).__init__()
        self.n_classes = n_classes
        self.n_features = n_features
        self.conv = nn.Sequential(nn.Conv1d(1, 3, 4),
                                     nn.MaxPool1d(3),
                                     nn.Conv1d(3, 5, 4),
                                     nn.MaxPool1d(3))
        # Вычисление размерности после применения сверток
        l = max_length
        for module in self.conv:
            l = np.ceil((l + 2 * module.padding - module.dilation * (module.kernel_size - 1)) / module.stride + 1)

        self.decoder = nn.Linear(n_features, n_classes)

    def forward(self, x):
        convolved = self.conv(x)
        pooled = torch.mean(convolved, dim=-1)
        decoded = self.decoder(pooled)
        probas = F.softmax(decoded, dim=-1)
        return probas


class ModelTraining:
    def __init__(self, model: nn.Module, optimizer, optimizer_params: dict, loss_fun, batch_size: int, log_dir: str,
                 scheduler=None, scheduler_params=None):
        self.use_gpu = torch.cuda.is_available()
        self.batch_size = batch_size
        self.model = model

        if self.use_gpu:
            self.model = self.model.cuda()
            # Распределяем вычисления на несколько GPU (если они у нас есть)
            # TODO: придется допилить iterate_minibatches,
            # чтобы после распараллеливания не получались батчи единичной длины, батчнорм перестает работать
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                self.model = nn.DataParallel(self.model)
        self.optimizer = optimizer(self.model.parameters(), **optimizer_params)
        if scheduler is not None:
            self.scheduler = scheduler(self.optimizer, **scheduler_params)
        else:
            self.scheduler = None
        self.loss_fun = loss_fun
        self.writer = SummaryWriter(log_dir)

        self.i = 0

    def train_epoch(self, inputs, targets):
        loss_log = []
        self.model.train()
        for idx in iterate_minibatches(inputs.shape[0], self.batch_size, shuffle=True):
            x_batch = inputs[idx, :]
            y_batch = targets[idx, :]
            x_batch, y_batch = to_torch(x_batch, y_batch, self.use_gpu)

            self.optimizer.zero_grad()
            res = self.model.forward(x_batch)

            loss = self.loss_fun(res, y_batch)
            loss.backward()
            self.optimizer.step()
            loss = loss.item()
            loss_log.append(loss)
            self.writer.add_scalar('loss/train_loss', loss, self.i)
            self.i += 1

        return loss_log

    def test(self, inputs, targets):
        loss_log = []
        self.model.eval()
        for idx in iterate_minibatches(inputs.shape[0], self.batch_size, shuffle=True):
            x_batch = inputs[idx, :]
            y_batch = targets[idx, :]
            x_batch, y_batch = to_torch(x_batch, y_batch, self.use_gpu)
            res = self.model.forward(x_batch)
            loss = self.loss_fun(res, y_batch)
            loss = loss.item()
            loss_log.append(loss)
        return loss_log

    def train(self, n_epochs, X_train, y_train, X_test, y_test):
        train_log = []
        val_log = []
        steps = 0
        for epoch in range(n_epochs):
            steps += X_train.shape[0] / self.batch_size
            train_loss = self.train_epoch(X_train, y_train)
            train_log.extend(train_loss)

            val_loss = self.test(X_test, y_test)
            if self.scheduler is not None:
                self.scheduler.step(np.mean(val_loss))
                for param_group in self.optimizer.param_groups:
                    lr = param_group['lr']
                self.writer.add_scalar('loss/lr', lr, self.i)
            val_log.append((steps, np.mean(val_loss)))
            self.writer.add_scalar('loss/val_loss', val_log[-1][1], val_log[-1][0])
            # В нынешнем варианте удается замерить только нулевые значения.
            '''
            if self.use_gpu:
                usages = parse_gpu_memory_usage()
                for i, u in enumerate(usages):
                    self.writer.add_scalar('memory_usage_GPU/memory_usage_GPU_%i'%i, u, self.i)
            '''

        return train_log, val_log

    # TODO: если будут проблемы с производительностью, можно будет выделить память разово
    # Нужно будет прогнать через модель какой-нибудь dummy sample, чтобы получить число классов
    def predict_proba(self, x):
        pred = []
        self.model.eval()
        for idx in iterate_minibatches(x.shape[0], self.batch_size, shuffle=False):
            x_batch, _ = to_torch(x[idx, :], np.zeros(2), self.use_gpu)
            res = self.model.forward(x_batch)
            pred.append(res.data.cpu().numpy())
        return np.vstack(pred)

    def serialize(self, path: str, method: str ='dict'):
        if method == 'dict':
            torch.save(self.model.state_dict(), path)
        else:
            torch.save(self.model, path)


def parse_gpu_memory_usage():
    cmd = subprocess.Popen(['nvidia-smi --query-gpu=utilization.memory --format=csv'],
                           shell=True,
                           stdout=subprocess.PIPE)
    stdout, stderr = cmd.communicate()
    usages = re.findall('(\d+)\s+%', str(stdout))
    return [float(_) for _ in usages]


# TODO: Типизация здесь может меняться в зависимости от задачи.
# y_batch надо делать FloatTensor только для специфического MultiLabelSoftMarginLoss
def to_torch(x_batch: Union[sparse.csr_matrix, np.ndarray], y_batch: np.ndarray, use_gpu: bool):
    """
    Функция, преобразующая вход в тензора pytorch, с опциональным перекладыванием на GPU
    :param x_batch: матрица признаков
    :param y_batch: матрица ответов. Если нужно преобразовать только x_batch, можно подавать любой ndarray
    :param use_gpu: нужно ли перекладывать на GPU.
    :return:
    """
    if isinstance(x_batch, sparse.csr_matrix):
        x_batch = x_batch.toarray()
    x_batch = torch.from_numpy(x_batch).type(torch.FloatTensor)

    y_batch = torch.from_numpy(y_batch).type(torch.FloatTensor)
    if use_gpu:
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()
    # return Variable(x_batch), Variable(y_batch)
    return x_batch, y_batch


def make_sparse_tensor(x: sparse.csr_matrix):
    rows, cols = x.nonzero()
    data = x.data
    i = torch.LongTensor([rows, cols])
    v = torch.FloatTensor(data)
    res = torch.sparse.FloatTensor(i, v, torch.Size(x.shape))
    return res


def iterate_minibatches(input_size: int, batchsize, shuffle=False):
    if shuffle:
        indices = np.random.permutation(input_size)
    for start_idx in range(0, input_size + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield excerpt


def get_examples(texts: List[str], vectorizer: Callable[[List[str]], np.ndarray], n: int, model: nn.Module,
                 labels: List[str], topk: int =20):
    queries = np.random.choice(texts, n)
    features = vectorizer(queries)
    inp = to_torch(features, np.zeros(2), torch.cuda.is_available())
    probas = model.forward(inp).data.cpu().numpy()
    top_idx = np.argsort(probas, axis=1)[:, :topk]
    predicted_labels = [[labels[i] for i in top_idx[j, :]] for j in top_idx.shape[0]]
    return queries, predicted_labels
