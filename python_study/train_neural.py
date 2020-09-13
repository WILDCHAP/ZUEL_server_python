# ����
import numpy as np
from two_layers_net import TwoLayersNet
from mnist import load_mnist
from time import *

begin_time = time()

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

#������
'''��������'''
item_size = 1000
'''��������'''
train_size = x_train.shape[0]
'''��������������'''
batch_size = 100
epoch = train_size / batch_size
learning_rate = 0.1

#��������
train_loss_list = []
train_acc_list = []
test_acc_list = []

network = TwoLayersNet(input_size=784, hidden_size=50, output_size=10)

for i in range(item_size):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(x_batch, t_batch)

    #����
    for key in network.params.keys():
        network.params[key] -= grads[key] * learning_rate

    #����
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    #print(loss)
    if i % epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("����������������", train_acc, " ����������������", test_acc)

end_time = time()
run_time = end_time-begin_time
print ('��������������������',run_time)