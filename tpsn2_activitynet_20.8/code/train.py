# @author: xiwu
# @license:
# @contact: fzy19931001@gmail.com
# @software: PyCharm
# @file: train.py
# @time: 2019/6/3 21:06
# @desc:

import numpy as np
import utils
import os
import random
from math import ceil
import torch
from torch.autograd import Variable
import torch.nn.functional as F

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Path & Hyperparameter
# NUM_VIDEOS = 4819
IMAGE_SIZE = 224
NUM_SEGMENTS = 400

INPUT_PATHS_thumos14 = {'train': {  # THUMOS 14 Validation Set
    'rgb': '../train_data/rgb_features',  # rgb
    'flow': '../train_data/flow_features',  # flow
},
    'test': {  # THUMOS 14 Test Set
        'rgb': '../test_data/rgb_features',  # rgb
        'flow': '../test_data/flow_features',  # flow
    }
}

INPUT_PATHS_activitynet1_2 = {'train': {  # THUMOS 14 Validation Set
    'rgb': '/home/huanbin/ActivityNet1.2_feat/train_data/rgb_features',  # rgb
    'flow': '/home/huanbin/ActivityNet1.2_feat/train_data/flow_features',  # flow
},
    'test': {  # THUMOS 14 Test Set
        'rgb': '/home/huanbin/ActivityNet1.2_feat/test_data/rgb_features',  # rgb
        'flow': '/home/huanbin/ActivityNet1.2_feat/test_data/flow_features',  # flow
    }
}

TRAIN_LABEL_PATH_thumos14 = '../train_data/train_labels.npy'

TRAIN_LABEL_PATH_activitynet1_2 = '/home/huanbin/ActivityNet1.2_feat/an_train_labels.npy'

device = torch.device("cuda")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def l1_penalty(var):
    return torch.abs(var).sum()


def train(sess, model, input_list, stream, train_iter, criterion, optimizer, criterion2, criterion3):
    dataset = input_list['dataset']
    erased_branch_num = input_list['erased_branch_num']
    batch_size = input_list['batch_size']
    beta = input_list['beta']
    learning_rate = input_list['learning_rate']
    ckpt = input_list['ckpt']

    if dataset == 'thumos14':
        INPUT_PATHS = INPUT_PATHS_thumos14
        TRAIN_LABEL_PATH = TRAIN_LABEL_PATH_thumos14
        NUM_VIDEOS = 200
        CLASSES = 20
    elif dataset == 'activitynet1.2':
        INPUT_PATHS = INPUT_PATHS_activitynet1_2
        TRAIN_LABEL_PATH = TRAIN_LABEL_PATH_activitynet1_2
        NUM_VIDEOS = 4819
        CLASSES = 100

    t_record = os.path.join(ckpt['path'], 'train_'+str(stream)+'_record.txt')
    f = open(t_record, 'w')

    label = np.load(TRAIN_LABEL_PATH)

    step_per_epoch = ceil(NUM_VIDEOS / batch_size)
    step = 1

    model.train()

    while step <= train_iter:
        shuffle_idx = random.sample(range(1, NUM_VIDEOS + 1), NUM_VIDEOS)
        for mini_step in range(step_per_epoch):
            # Get mini batch index (batch_size)
            minibatch_index = shuffle_idx[mini_step * batch_size: (mini_step + 1) * batch_size]
            # minibatch = utils.processVid(minibatch_index, INPUT_PATHS['train'][stream],
            #                              NUM_SEGMENTS)  # Return [batch_size, segment nums, 16, 224, 224, 3]
            minibatch_label = label[minibatch_index - np.ones((len(minibatch_index),), dtype=int)].astype(np.float32)
            # minibatch_input = minibatch.reshape(len(minibatch_index) * NUM_SEGMENTS, 1024).astype(np.float32)
            minibatch_input = np.load(os.path.join(INPUT_PATHS['train'][stream], '{:d}.npy'.format(minibatch_index[0]))).astype(np.float32)

            labels = []
            for ml in range(CLASSES):
                if minibatch_label[0, ml] == 1:
                    labels.append(ml)
            # labels = torch.from_numpy(np.array(labels)).to(device)
            labels = Variable(torch.from_numpy(np.array(labels)).to(device))

            # minibatch_input = torch.from_numpy(minibatch_input).float().to(device)
            # minibatch_label = torch.from_numpy(minibatch_label).float().to(device)
            minibatch_input = Variable(torch.from_numpy(minibatch_input).float().to(device))
            minibatch_label = Variable(torch.from_numpy(minibatch_label).float().to(device))

            outputs, attention, tcam, erased_outputs, erased_tcam, erased_att, erased2_outputs, erased2_tcam, erased2_att = \
                model(minibatch_input, labels, is_training=True)

            train_loss = criterion(outputs, minibatch_label)  # outputs.size: torch.Size([20]), mi._label.size: [1, 20]
            branch2_loss = criterion2(erased_outputs, minibatch_label)
            branch3_loss = criterion3(erased2_outputs, minibatch_label)
            l11 = l1_penalty(attention) / 5000
            l12 = l1_penalty(erased_att) / 5000
            l13 = l1_penalty(erased2_att) / 5000
            # l11 = l1_penalty(attention) / 5500
            # l12 = l1_penalty(erased_att) / 5500
            # l13 = l1_penalty(erased2_att) / 5500
            if erased_branch_num == 0:
                train_loss = train_loss + l11
            elif erased_branch_num == 1:
                train_loss = train_loss + branch2_loss + l11 + l12
            elif erased_branch_num == 2:
                train_loss = train_loss + branch2_loss + branch3_loss + l11 + l12 + l13

            step += 1

            cou = outputs.cpu().data.numpy()
            ep = erased_outputs.cpu().data.numpy()
            lab = minibatch_label.cpu().data.numpy()
            vie = attention.view(-1).cpu().data.numpy()
            tcam = tcam.cpu().data.numpy()
            stcam = sigmoid(tcam)
            erased_tcam = erased_tcam.cpu().data.numpy()
            serased_tcam = sigmoid(erased_tcam)
            # vie = attention.view(-1, NUM_SEGMENTS).cpu().data.numpy()

            optimizer.zero_grad()
            train_loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            # Print the loss and save weights after every 100 iteration
            if step % 600 == 0:
                # train_loss = sess.run(model.loss,
                #                       feed_dict={model.X: minibatch_input, model.Y: minibatch_label, model.BETA: beta,
                #                                  model.LEARNING_RATE: learning_rate})
                print('iter {:d}, {} train loss {:g}'.format(step, stream, train_loss))
                f.write('iter {:d}, {} train loss {:g}\n'.format(step, stream, train_loss))
                f.flush()

                torch.save({"model": model.state_dict()}, os.path.join(ckpt[stream],
                                                                       '{:s}_{:d}.pth'.format(stream, step)))

                print('Checkpoint {:d} Saved'.format(step))
    f.close()
