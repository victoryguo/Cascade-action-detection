# @author: xiwu
# @license:
# @contact: fzy19931001@gmail.com
# @software: PyCharm
# @file: utils.py
# @time: 2019/6/3 21:07
# @desc:

import os
import sys
import numpy as np
import argparse
import tensorflow as tf
from scipy.interpolate import interp1d
from nms_cpu import nms_cpu

NUM_SEGMENTS = 400
SAMPLING_FRAMES = 25
NUM_INPUT_FRAMES = 16
# NUM_CLASS = 100

CLASS_thumos14 = {0: 'BaseballPitch', 1: 'BasketballDunk', 2: 'Billiards', 3: 'CleanAndJerk', 4: 'CliffDiving',
                  5: 'CricketBowling', 6: 'CricketShot', 7: 'Diving', 8: 'FrisbeeCatch', 9: 'GolfSwing',
                  10: 'HammerThrow', 11: 'HighJump', 12: 'JavelinThrow', 13: 'LongJump', 14: 'PoleVault',
                  15: 'Shotput', 16: 'SoccerPenalty', 17: 'TennisSwing', 18: 'ThrowDiscus', 19: 'VolleyballSpiking'}

CLASS_activitynet1_2 = {0: 'Rock climbing', 1: 'Drinking beer', 2: 'Vacuuming floor', 3: 'Dodgeball', 4: 'Paintball',
         5: 'Playing field hockey', 6: 'Washing hands', 7: 'Doing karate', 8: 'Playing volleyball',
         9: 'Playing violin', 10: 'Horseback riding', 11: 'Shaving legs', 12: 'Grooming horse',
         13: 'Preparing salad', 14: 'Windsurfing', 15: 'Skateboarding', 16: 'Spinning', 17: 'Cricket',
         18: 'Smoking a cigarette', 19: 'Hand washing clothes', 20: 'Doing step aerobics', 21: 'Removing curlers',
         22: 'Doing motocross', 23: 'Brushing hair', 24: 'Washing face', 25: 'Long jump', 26: 'Getting a piercing',
         27: 'Hammer throw', 28: 'Shot put', 29: 'Kayaking', 30: 'Putting on makeup', 31: 'Plataform diving',
         32: 'Javelin throw', 33: 'Mixing drinks', 34: 'Zumba', 35: 'Playing saxophone',
         36: 'Layup drill in basketball', 37: 'Tennis serve with ball bouncing', 38: 'Cleaning windows',
         39: 'Playing flauta', 40: 'Playing harmonica', 41: 'Getting a haircut', 42: 'Cheerleading',
         43: 'Using the balance beam', 44: 'Tango', 45: 'Springboard diving', 46: 'Playing water polo',
         47: 'Doing nails', 48: 'Getting a tattoo', 49: 'Wrapping presents', 50: 'Pole vault', 51: 'Tai chi',
         52: 'Making a sandwich', 53: 'Cumbia', 54: 'Shaving', 55: 'Playing lacrosse', 56: 'Painting',
         57: 'Belly dance', 58: 'Snatch', 59: 'Ironing clothes', 60: 'Drinking coffee', 61: 'Discus throw',
         62: 'Doing kickboxing', 63: 'Playing polo', 64: 'Chopping wood', 65: 'Walking the dog',
         66: 'Using parallel bars', 67: 'Archery', 68: 'Mowing the lawn', 69: 'Playing badminton',
         70: 'Shoveling snow', 71: 'Washing dishes', 72: 'Fixing bicycle', 73: 'Smoking hookah',
         74: 'Polishing shoes', 75: 'Playing kickball', 76: 'Using the pommel horse', 77: 'Hopscotch',
         78: 'Bungee jumping', 79: 'Clean and jerk', 80: 'Playing piano', 81: 'Sailing', 82: 'Preparing pasta',
         83: 'Playing bagpipes', 84: 'Playing racquetball', 85: 'Cleaning shoes', 86: 'Triple jump',
         87: 'Polishing forniture', 88: 'Brushing teeth', 89: 'Ballet', 90: 'Playing squash', 91: 'High jump',
         92: 'Bathing dog', 93: 'Playing guitarra', 94: 'Ping-pong', 95: 'Breakdancing', 96: 'Using uneven bars',
         97: 'Tumbling', 98: 'Playing accordion', 99: 'Starting a campfire'}

CLASS_INDEX = {'BaseballPitch': '7', 'BasketballDunk': '9', 'Billiards': '12', 'CleanAndJerk': '21', 'CliffDiving': '22', 'CricketBowling': '23', 'CricketShot': '24', 'Diving': '26', 'FrisbeeCatch': '31', 'GolfSwing': '33',
         'HammerThrow': '36', 'HighJump': '40', 'JavelinThrow': '45', 'LongJump': '51', 'PoleVault': '68', 'Shotput': '79', 'SoccerPenalty': '85', 'TennisSwing': '92', 'ThrowDiscus': '93', 'VolleyballSpiking': '97'}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='test', type=str)
    # input for training
    parser.add_argument('--training_num', default=303000, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--beta', default=0.0001, type=float)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--ckpt', default=1, type=int)
    # input for inference
    parser.add_argument('--test_iter', default=164400, type=int)
    parser.add_argument('--test_iter2', default=142800, type=int)
    parser.add_argument('--class_th', default=0.1, type=float)
    parser.add_argument('--scale', default=24, type=int)
    # huanbin add
    parser.add_argument('--dataset', type=str, default='thumos14', help='thumos14 or activitynet1.2')
    parser.add_argument('--train_stream', type=str, default='rgb', help='rgb or flow stream used to train')
    parser.add_argument('--test_mode', type=str, default='fusion', help='stream used to train -- fusion(rgb&flow), rgb, flow')
    parser.add_argument('--erased_branch_num', type=int, default=1, help='how much erased branch in the network -- 0, 1, 2')
    args = parser.parse_args()
    return args


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ckpt_path(c):
    directory = os.path.join('ckpt', 'ckpt{:03d}'.format(c))
    if not os.path.exists(directory):
        os.makedirs(directory)
        os.makedirs(os.path.join(directory, 'rgb'))
        os.makedirs(os.path.join(directory, 'flow'))

    cp = dict(path=directory, rgb=os.path.join(directory, 'rgb'),
              flow=os.path.join(directory, 'flow'))
    return cp


def inf_progress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def random_perturb(v_len):
    random_p = np.arange(NUM_SEGMENTS) * v_len / NUM_SEGMENTS
    for i in range(NUM_SEGMENTS):
        if i < NUM_SEGMENTS - 1:
            if int(random_p[i]) != int(random_p[i + 1]):
                random_p[i] = np.random.choice(range(int(random_p[i]), int(random_p[i + 1]) + 1))
            else:
                random_p[i] = int(random_p[i])
        else:
            if int(random_p[i]) < v_len - 1:
                random_p[i] = np.random.choice(range(int(random_p[i]), v_len))
            else:
                random_p[i] = int(random_p[i])
    return random_p.astype(int)


def uniform_sampling(vid_len):
    u_sample = np.arange(NUM_SEGMENTS) * vid_len / NUM_SEGMENTS
    u_sample = np.floor(u_sample)
    return u_sample.astype(int)


# Process training data
def processVid(idx, f_path, numSeg):
    batch_frames = np.zeros((len(idx), numSeg, 1024))
    for i in range(len(idx)):
        numvid = idx[i]
        feature_path = os.path.join(f_path, '{:d}.npy'.format(numvid))
        feature = np.load(feature_path).astype(np.float32)
        seg_list = random_perturb(feature.shape[0])
        batch_frames[i] = feature[seg_list]

    return batch_frames


# Process test data
def processTestVid(idx, fpath, numSeg):
    rgb_frames = np.zeros((numSeg, 1024))
    flow_frames = np.zeros((numSeg, 1024))

    rgbpath = os.path.join(fpath['rgb'], '{:d}.npy'.format(idx))
    flowpath = os.path.join(fpath['flow'], '{:d}.npy'.format(idx))

    rvid = np.load(rgbpath).astype(np.float32)
    fvid = np.load(flowpath).astype(np.float32)

    seg_list = uniform_sampling(rvid.shape[0])

    rgb_frames = rvid[seg_list]
    flow_frames = fvid[seg_list]

    return rgb_frames, flow_frames, seg_list, rvid.shape[0]


# Localization functions (post-processing)
# Get TCAM signal
def get_tCAM(feature, layer_Weights):
    tCAM = np.matmul(feature, layer_Weights)
    return tCAM


# Get weighted TCAM and the score for each segment
def get_wtCAM(main_tCAM, sub_tCAM, attention_Weights, alpha, pred):
    # attention_Weights = attention_Weights.cpu().data.numpy()
    wtCAM = sigmoid(main_tCAM)
    sub_tCAM = sigmoid(sub_tCAM)
    signal = np.reshape(wtCAM[:, pred], (attention_Weights.shape[0], -1, 1))
    score = np.reshape((alpha * main_tCAM + (1 - alpha) * sub_tCAM)[:, pred],
                       (attention_Weights.shape[0], -1, 1))
    ress = np.concatenate((signal, score), axis=2)
    return ress


# Interpolate empty segments
def upgrade_resolution(arr, scale):
    x = np.arange(0, arr.shape[0])
    f = interp1d(x, arr, kind='linear', axis=0, fill_value='extrapolate')  # linear/quadratic/cubic
    scale_x = np.arange(0, arr.shape[0], 1 / scale)
    up_scale = f(scale_x)
    return up_scale


# Interpolate the wtCAM signals and threshold
def interpolated_wtCAM(wT, scale):
    final_wT = upgrade_resolution(wT, scale)
    result_zero = np.where(final_wT[:, :, 0] < 0.05)
    final_wT[result_zero] = 0
    return final_wT


# Return the index where the wtcam value > 0.05
def get_tempseg_list(wtcam, c_len):
    args = parse_args()
    temp = []
    for i in range(c_len):
        if args.dataset == 'thumos14':
            pos = np.where(wtcam[:, i, 0] > 0.5024)
        elif args.dataset == 'activitynet1.2':
            pos = np.where(wtcam[:, i, 0] > 0.4)
        temp_list = pos
        temp.append(temp_list)
    return temp


# Group the connected results
def grouping(arr):
    return np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)


# Get the temporal proposal
def get_temp_proposal(tList, wtcam, c_pred, scale, v_len):
    # t_factor = (NUM_INPUT_FRAMES * v_len) / (scale * NUM_SEGMENTS * SAMPLING_FRAMES)
    # Factor to convert segment index to actual timestamp
    t_factor = 16.0 / 25.0
    temp = []
    for i in range(len(tList)):
        c_temp = []
        temp_list = np.array(tList[i])[0]
        if temp_list.any():
            grouped_temp_list = grouping(temp_list)  # Get the connected parts
            for j in range(len(grouped_temp_list)):
                if grouped_temp_list[j].shape[0] == 1:
                    continue
                c_score = np.mean(wtcam[grouped_temp_list[j], i, 1])
                t_start = grouped_temp_list[j][0] * t_factor
                t_end = (grouped_temp_list[j][-1] + 1) * t_factor
                c_temp.append([c_pred[i], c_score, t_start, t_end])  # Add the proposal
        temp.append(c_temp)
    return temp


# Perform Non-Maximum-Suppression
def nms_prop(arr):
    args = parse_args()
    if args.dataset == 'thumos14':
        nms_thres = 0.95
    elif args.dataset == 'activitynet1.2':
        nms_thres = 0.7
    classes = list(set(list(arr[:, 0])))
    ress = []
    for i in range(len(classes)):
        tmp_cls = arr[arr[:, 0] == classes[i]]
        keeps = nms_cpu(tmp_cls, nms_thres)
        for ke in keeps:
            ress.append(tmp_cls[ke])
    return ress


# Fuse two stream & perform non-maximum suppression
def integrated_prop(rgbProp, flowProp, rPred, fPred):
    args = parse_args()
    if args.dataset == 'thumos14':
        NUM_CLASS = 20
    elif args.dataset == 'activitynet1.2':
        NUM_CLASS = 100
    temp = []
    for i in range(NUM_CLASS):
        if (i in rPred) and (i in fPred):
            ridx = rPred.index(i)
            fidx = fPred.index(i)
            rgb_temp = rgbProp[ridx]
            flow_temp = flowProp[fidx]
            rgb_set = set([tuple(x) for x in rgb_temp])
            flow_set = set([tuple(x) for x in flow_temp])
            fuse_temp = np.array([x for x in rgb_set | flow_set])  # Gather RGB proposals and FLOW proposals together
            fuse_temp = np.sort(fuse_temp.view('f8,f8,f8,f8'), order=['f1'], axis=0).view(np.float)[::-1]

            if len(fuse_temp) > 0:
                ress = nms_prop(fuse_temp)
                for k in ress:
                    temp.append(k)

        elif (i in rPred) and not (i in fPred):  # For the video which only has RGB Proposals
            ridx = rPred.index(i)
            rgb_temp = rgbProp[ridx]
            for j in range(len(rgb_temp)):
                temp.append(rgb_temp[j])
        elif not (i in rPred) and (i in fPred):  # For the video which only has FLOW Proposals
            fidx = fPred.index(i)
            flow_temp = flowProp[fidx]
            for j in range(len(flow_temp)):
                temp.append(flow_temp[j])
    return temp


# Record the proposals to the json file
def result2json(result):
    args = parse_args()
    if args.dataset == 'thumos14':
        CLASS = CLASS_thumos14
    elif args.dataset == 'activitynet1.2':
        CLASS = CLASS_activitynet1_2
    result_file = []
    for i in range(len(result)):
        for j in range(len(result[i])):
            line = {'label': CLASS[result[i][j][0]], 'score': result[i][j][1],
                    'segment': [result[i][j][2], result[i][j][3]]}
            result_file.append(line)
    return result_file


def json2txt(jF, rF):
    for i in jF.keys():
        for j in range(len(jF[i])):
            rF.write('{:s} {:f} {:f} {:s} {:f}\n'.format(i, jF[i][j]['segment'][0], jF[i][j]['segment'][1],
                                                        CLASS_INDEX[jF[i][j]['label']], round(jF[i][j]['score'], 6)))
