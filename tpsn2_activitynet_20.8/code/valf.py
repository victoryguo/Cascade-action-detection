# @author: xiwu
# @license:
# @contact: fzy19931001@gmail.com
# @software: PyCharm
# @file: valf.py
# @time: 2019/6/6 10:13
# @desc:

import numpy as np
import utils
import os
import json
import torch

device = torch.device("cuda")


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

# Path & Hyperparameter
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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_wtCAM(main_tCAM, sub_tCAM, attention_Weights, alpha, pred):
    wtCAM = sigmoid(main_tCAM)
    signal = np.reshape(wtCAM[:, pred], (attention_Weights.shape[0], -1, 1))
    if sub_tCAM == "":
        score = np.reshape(main_tCAM[:, pred],
                           (attention_Weights.shape[0], -1, 1))
    else:
        score = np.reshape(main_tCAM[:, pred],
                           (attention_Weights.shape[0], -1, 1))
    ress = np.concatenate((signal, score), axis=2)
    return ress


def valf(sess, model, init, input_list, test_iter):
    dataset = input_list['dataset']
    ckpt = input_list['ckpt']
    scale = input_list['scale']
    class_threshold = input_list['class_threshold']

    if dataset == 'thumos14':
        INPUT_PATHS = INPUT_PATHS_thumos14
        TEST_NUM = 210
        ALPHA = 0.5
        test_vid_list = open('THUMOS14_test_vid_list.txt', 'r')  # file for matching 'video number' and 'video name'
    elif dataset == 'activitynet1.2':
        INPUT_PATHS = INPUT_PATHS_activitynet1_2
        TEST_NUM = 2383
        ALPHA = 0.5
        test_vid_list = open('AN_test_vid_list.txt', 'r')  # file for matching 'video number' and 'video name'

    lines = test_vid_list.read().splitlines()

    # Define json File (output)
    final_result = {}
    final_result['version'] = 'VERSION 1.3'
    final_result['results'] = {}
    final_result['external_data'] = {'used': True, 'details': 'Features from I3D Net'}

    for i in range(1, TEST_NUM + 1):
        vid_name = lines[i - 1]
        # Load Frames
        rgb_path = os.path.join(INPUT_PATHS['test']['rgb'], '{:d}.npy'.format(i))
        rgb_features = np.load(rgb_path).astype(np.float32)
        rgb_features = rgb_features.astype(np.float32)

        rgb_features = torch.from_numpy(rgb_features).float().to(device)

        # RGB Stream
        model.load_state_dict(torch.load(os.path.join(ckpt['rgb'], 'rgb_' + str(test_iter) + '.pth'))["model"])
        model.eval()

        # rgb_class_result, rgb_attention, rgb_tCam, erased_outputs, erased_tcam, erased_att = model(rgb_features, "")
        rgb_class_result, rgb_attention, rgb_tCam, \
        erased_outputs, erased_tcam, erased_att, \
        erased2_outputs, erased2_tcam, erased2_att = model(rgb_features, "")
        rgb_tCam = rgb_tCam.cpu().data.numpy()

        # min_tcam = np.min(rgb_tCam, 0)
        # max_tcam = np.max(rgb_tCam, 0)
        # trgbcam = (rgb_tCam - min_tcam) / (max_tcam - min_tcam)

        erased_tcam = erased_tcam.cpu().data.numpy()

        rgb_class_result = rgb_class_result.cpu().data.numpy()
        erased_outputs = erased_outputs.cpu().data.numpy()
        # rgb_class_result = np.maximum(rgb_class_result, erased_outputs)
        rgb_attention = rgb_attention.cpu().data.numpy()
        erased_att = erased_att.cpu().data.numpy()
        fusion_attention = np.maximum(rgb_attention, erased_att)

        # -------- test ------------------------
        sig_rgb = sigmoid(rgb_tCam)
        sig_era = sigmoid(erased_tcam)
        sig_att_rgb = rgb_attention * sig_rgb
        sig_att_era = erased_att * sig_era
        # -------- test ------------------------

        # emin_tcam = np.min(erased_tcam, 0)
        # emax_tcam = np.max(erased_tcam, 0)
        # teracam = (erased_tcam - emin_tcam) / (emax_tcam - emin_tcam)

        fusion_tCam = np.maximum(rgb_tCam, erased_tcam)

        # sig_fusion_tCam = sigmoid(fusion_tCam)
        # sig_fu_att_tCam = rgb_attention * sig_fusion_tCam
        # sig_r_t = rgb_attention * sig_rgb

        rgb_attention = fusion_attention
        rgb_tCam = fusion_tCam

        # Gathering Classification Result
        rgb_class_prediction = np.where(rgb_class_result > class_threshold)[0]
        if not rgb_class_prediction.any():
            maxind = np.argmax(rgb_class_result)
            maxnp = np.array([maxind])
            rgb_class_prediction = maxnp

        if rgb_class_prediction.any():
            # Weighted T-CAM
            rgb_wtCam = get_wtCAM(rgb_tCam, "", rgb_attention, ALPHA, rgb_class_prediction)
            # Get segment list of rgb_int_wtCam
            rgb_temp_idx = utils.get_tempseg_list(rgb_wtCam, len(rgb_class_prediction))
            # Temporal Proposal
            rgb_temp_prop = utils.get_temp_proposal(rgb_temp_idx, rgb_wtCam, rgb_class_prediction,
                                                    scale, rgb_class_result.shape[0])

        final_result['results'][vid_name] = utils.result2json(rgb_temp_prop)

        utils.inf_progress(i, TEST_NUM, 'Progress', 'Complete', 1, 50)

    # Save Results
    json_path = os.path.join(ckpt['path'], 'results.json')
    with open(json_path, 'w') as fp:
        json.dump(final_result, fp, cls=MyEncoder)

    # txt_path = os.path.join(ckpt['path'], 'results.txt')
    # with open(txt_path, 'w') as tp:
    #     utils.json2txt(final_result['results'], tp)

    test_vid_list.close()
