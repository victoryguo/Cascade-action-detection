# @author: xiwu
# @license:
# @contact: fzy19931001@gmail.com
# @software: PyCharm
# @file: val.py
# @time: 2019/6/4 14:56
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
NUM_SEGMENTS = 400  # Should be larger : 400

INPUT_PATHS_thumos14 = {'train': {  # THUMOS 14 Validation Set
    'rgb': '../train_data/rgb_features',  # rgb
    'flow': '../train_data/flow_features',  # flow
},
    'test': {  # THUMOS 14 Test Set
        'rgb': '../test_data/rgb_features',  # rgb
        'flow': '../test_data/flow_features',  # flow
    }
}

INPUT_PATHS_activitynet1_2 = {'train': {
    'rgb': '/home/huanbin/ActivityNet1.2_feat/train_data/rgb_features',  # rgb
    'flow': '/home/huanbin/ActivityNet1.2_feat/train_data/flow_features',  # flow
},
    'test': {
        'rgb': '/home/huanbin/ActivityNet1.2_feat/test_data/rgb_features',  # rgb
        'flow': '/home/huanbin/ActivityNet1.2_feat/test_data/flow_features',  # flow
    }
}


def val(sess, model, init, input_list, test_iter, model2, test_iter2):
    ckpt = input_list['ckpt']
    scale = input_list['scale']
    class_threshold = input_list['class_threshold']
    dataset = input_list['dataset']
    erased_branch_num = input_list['erased_branch_num']

    if dataset == 'thumos14':
        INPUT_PATHS = INPUT_PATHS_thumos14
        # I excluded two falsely annotated videos, 270, 1496, following the SSN paper (https://arxiv.org/pdf/1704.06228.pdf)
        TEST_NUM = 210
        ALPHA = 0.7
        test_vid_list = open('THUMOS14_test_vid_list.txt', 'r') # file for matching 'video number' and 'video name'
    elif dataset == 'activitynet1.2':
        INPUT_PATHS = INPUT_PATHS_activitynet1_2
        TEST_NUM = 2383
        ALPHA = 0.5
        test_vid_list = open('AN_test_vid_list.txt', 'r') # file for matching 'video number' and 'video name'

    lines = test_vid_list.read().splitlines()

    # Define json File (output)
    final_result = {}
    final_result['version'] = 'VERSION 1.3'
    final_result['results'] = {}
    final_result['external_data'] = {'used': True, 'details': 'Features from I3D Net'}

    for i in range(1, TEST_NUM + 1):
        vid_name = lines[i - 1]
        # Load Frames
        # rgb_features, flow_features, temp_seg, vid_len = utils.processTestVid(i, INPUT_PATHS['test'],
        #                                                                       NUM_SEGMENTS)
        rgb_path = os.path.join(INPUT_PATHS['test']['rgb'], '{:d}.npy'.format(i))
        rgb_features = np.load(rgb_path).astype(np.float32)
        rgb_features = rgb_features.astype(np.float32)

        flow_path = os.path.join(INPUT_PATHS['test']['flow'], '{:d}.npy'.format(i))
        flow_features = np.load(flow_path).astype(np.float32)
        flow_features = flow_features.astype(np.float32)

        rgb_features = torch.from_numpy(rgb_features).float().to(device)
        flow_features = torch.from_numpy(flow_features).float().to(device)

        # RGB Stream
        model.load_state_dict(torch.load(os.path.join(ckpt['rgb'], 'rgb_' + str(test_iter) + '.pth'))["model"])
        model.eval()

        model2.load_state_dict(torch.load(os.path.join(ckpt['flow'], 'flow_' + str(test_iter2) + '.pth'))["model"])
        model2.eval()

        rgb_class_w = list(model.parameters())[-2].cpu().data.numpy().T
        flow_class_w = list(model2.parameters())[-2].cpu().data.numpy().T

        rgb_class_result, rgb_attention, rgb_tCam, \
        rgb_erased_outputs, rgb_erased_tcam, rgb_erased_att, \
        rgb_erased2_outputs, rgb_erased2_tcam, rgb_erased2_att = model(rgb_features, "")
        flow_class_result, flow_attention, flow_tCam, \
        flow_erased_outputs, flow_erased_tcam, flow_erased_att, \
        flow_erased2_outputs, flow_erased2_tcam, flow_erased2_att = model2(flow_features, "")

        rgb_tcam = rgb_tCam.cpu().data.numpy()
        flow_tcam = flow_tCam.cpu().data.numpy()
        rgb_erased_tcam = rgb_erased_tcam.cpu().data.numpy()
        flow_erased_tcam = flow_erased_tcam.cpu().data.numpy()
        rgb_erased2_tcam = rgb_erased2_tcam.cpu().data.numpy()
        flow_erased2_tcam = flow_erased2_tcam.cpu().data.numpy()
        if erased_branch_num == 0:
            rgb_tCam = rgb_tcam
            flow_tCam = flow_tcam
        elif erased_branch_num == 1:
            rgb_tCam = np.maximum(rgb_tcam, rgb_erased_tcam)
            flow_tCam = np.maximum(flow_tcam, flow_erased_tcam)
        elif erased_branch_num == 2:
            rgb_tCam_1 = np.maximum(rgb_tcam, rgb_erased_tcam)
            rgb_tCam = np.maximum(rgb_tCam_1, rgb_erased2_tcam)
            flow_tCam_1 = np.maximum(flow_tcam, flow_erased_tcam)
            flow_tCam = np.maximum(flow_tCam_1, flow_erased2_tcam)

        rgb_class_result = rgb_class_result.cpu().data.numpy()
        flow_class_result = flow_class_result.cpu().data.numpy()
        zzz2 = rgb_attention.cpu().data.numpy()

        # Gathering Classification Result
        rgb_class_prediction = np.where(rgb_class_result > class_threshold)[0]
        flow_class_prediction = np.where(flow_class_result > class_threshold)[0]

        if not rgb_class_prediction.any():
            maxind = np.argmax(rgb_class_result)
            maxnp = np.array([maxind])
            rgb_class_prediction = maxnp
        if not flow_class_prediction.any():
            maxind = np.argmax(flow_class_result)
            maxnp = np.array([maxind])
            flow_class_prediction = maxnp

        r_check = False
        f_check = False
        if rgb_class_prediction.any():
            r_check = True
            # Weighted T-CAM
            rgb_wtCam = utils.get_wtCAM(rgb_tCam, flow_tCam, rgb_attention, ALPHA, rgb_class_prediction)
            # Interpolate W-TCAM
            # rgb_int_wtCam = utils.interpolated_wtCAM(rgb_wtCam, scale)
            # Get segment list of rgb_int_wtCam
            rgb_temp_idx = utils.get_tempseg_list(rgb_wtCam, len(rgb_class_prediction))
            # Temporal Proposal
            rgb_temp_prop = utils.get_temp_proposal(rgb_temp_idx, rgb_wtCam, rgb_class_prediction,
                                                    scale, rgb_class_result.shape[0])

        if flow_class_prediction.any():
            f_check = True
            # Weighted T-CAM
            flow_wtCam = utils.get_wtCAM(flow_tCam, rgb_tCam, flow_attention, 1 - ALPHA, flow_class_prediction)
            # Get segment list of rgb_int_wtCam
            flow_temp_idx = utils.get_tempseg_list(flow_wtCam, len(flow_class_prediction))
            # Temporal Proposal
            flow_temp_prop = utils.get_temp_proposal(flow_temp_idx, flow_wtCam, flow_class_prediction,
                                                    scale, flow_class_result.shape[0])

        # final_result['results'][vid_name] = utils.result2json(flow_temp_prop)

        if r_check and f_check:
            # Fuse two stream and perform non-maximum suppression
            temp_prop = utils.integrated_prop(rgb_temp_prop, flow_temp_prop, list(rgb_class_prediction),
                                              list(flow_class_prediction))
            final_result['results'][vid_name] = utils.result2json([temp_prop])
        elif r_check and not f_check:
            final_result['results'][vid_name] = utils.result2json(rgb_temp_prop)
        elif not r_check and f_check:
            final_result['results'][vid_name] = utils.result2json(flow_temp_prop)

        utils.inf_progress(i, TEST_NUM, 'Progress', 'Complete', 1, 50)

    # Save Results
    json_path = os.path.join(ckpt['path'], 'hb_check_results.json')
    with open(json_path, 'w') as fp:
        json.dump(final_result, fp, cls=MyEncoder)

    # txt_path = os.path.join(ckpt['path'], 'results.txt')
    # with open(txt_path, 'w') as tp:
    #     utils.json2txt(final_result['results'], tp)

    test_vid_list.close()
