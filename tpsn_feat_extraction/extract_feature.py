# Copyright 2018 Jae Yoo Park
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import numpy as np
import tensorflow as tf
import os
import utils_feature as uf
import argparse
import cv2

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

IMAGE_SIZE = 224
INPUT_VIDEO_FRAMES = 16  # 每16帧为一个segment
NUM_VIDEOS = {'train': 200, 'test': 210}
# video path
INPUT_PATHS = {'train': {'rgb': '/home/fzy/THUMOS14/train_frames/',  # rgb frames
                         'flow': '/home/fzy/THUMOS14/train_frames/',  # flow frames
                         },
               'test': {'rgb': '/home/fzy/THUMOS14/test_frames/',  # rgb  frames
                        'flow': '/home/fzy/THUMOS14/test_frames/',  # flow frames
                        }
               }
SAVE_PATHS = {'train': {'rgb': './train_data/rgb_features',  # rgb feat
                        'flow': './train_data/flow_features',  # flow feat
                        },
              'test': {'rgb': './test_data/rgb_features',  # rgb feat
                       'flow': './test_data/flow_features',  # flow feat
                       }
              }
# INPUT_PATHS = {'train': {'rgb': '../train_data/rgb',  # rgb frames
#                          'flow': '../train_data/flows',  # flow
#                          },
#                'test': {'rgb': '../test_data/rgb',  # rgb  frames
#                         'flow': '../test_data/flows',  # flow
#                         }
#                }
# SAVE_PATHS = {'train': {'rgb': '../train_data/rgb_features',  # rgb
#                         'flow': '../train_data/flow_features',  # flow
#                         },
#               'test': {'rgb': '../test_data/rgb_features',  # rgb
#                        'flow': '../test_data/flow_features',  # flow
#                        }
#               }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream', default='rgb', type=str)
    parser.add_argument('--data', default='train', type=str)
    argms = parser.parse_args()
    return argms

# Main : extract features
args = parse_args()
stream = args.stream
data = args.data

init = tf.global_variables_initializer()

if data == 'train':
    if stream == 'rgb':
        data_path = r'/home/fzy/THUMOS14/train_frames'
    elif stream == 'flow':
        data_path = r'/home/huanbin/myproject/tpsn_feat_extraction/flow_frames/flow_train'
elif data == 'test':
    if stream == 'rgb':
        data_path = r'/home/fzy/THUMOS14/test_frames'
    elif stream == 'flow':
        data_path = r'/home/huanbin/myproject/tpsn_feat_extraction/flow_frames/flow_train'

print('data_path: ', data_path)
video_name = os.listdir(data_path)
video_name = sorted(video_name)  # 按视频名称最后的编号排序，这样方便对应提取train_labels


if data == 'test':
    video_name = []
    f = open('./THUMOS14_test_vid_list.txt')
    lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i]
        vid_name = line.strip('\n')
        video_name.append(vid_name)

print('video_name: ', video_name)

# if data == 'test':
#     # I excluded two falsely annotated videos, 270, 1496, following the SSN paper (https://arxiv.org/pdf/1704.06228.pdf)
#     video_name.remove('video_test_0000270')
#     video_name.remove('video_test_0001496')
#     video_name.remove('video_test_0001558')  # 删除最后一个，凑齐210

# 利用for循环逐个视频提取特征，这里作者简单把每一个视频剪切的帧保存的文件夹命名为1、2、3...
for i in range(1, NUM_VIDEOS[data] + 1):
    # vid_path = os.path.join(INPUT_PATHS[data][stream], '{:d}'.format(i))  # path -- 查找到对应的某一个视频文件夹
    vid_path = os.path.join(INPUT_PATHS[data][stream], video_name[i-1])  # path -- 查找到对应的某一个视频文件夹
    print('video path: ', vid_path)
    num_vid_frame = len(os.listdir(vid_path))  # number of total frames -- 某一个视频帧文件夹下的所有图片
    #  每16帧1个特征向量，因此总分段将是（总帧数/ 16）
    num_segments = int(num_vid_frame / INPUT_VIDEO_FRAMES)  # number of total segments (ex. 1 feature vector per 16 frames, so total segments will be (total frames / 16))
    # Load feature model
    print('model {:d} loaded'.format(i))
    # 打印某个视频对应总共有多少帧、一共可以被分为多少个segment
    print('{:d} : {:d} frames, {:d} segs'.format(i, num_vid_frame, num_segments))
    X_feature = np.zeros((num_segments, 1024))
    # num_hundred_segments = int(num_segments / 100)   # 作者一次处理100个segment -- 这个要根据自己的GUP实际内存自行调整！！！
    num_hundred_segments = int(num_segments / 50)  # 我的GPU内存不够，所以这里暂且设置为50好了~_~
    # Load Images
    channel = 3
    vid = np.zeros((num_vid_frame, IMAGE_SIZE, IMAGE_SIZE, channel))  # 单个视频的维度
    # 逐帧读取某一个视频的所有帧，并得到其numpy格式的张量数据
    for frame in range(num_vid_frame):
        frm = os.path.join(vid_path, '{:06d}.png'.format(frame))
        vid[frame] = cv2.imread(frm, cv2.IMREAD_COLOR)
    print('{:d} vid frames loaded'.format(i))  # 输出第几个视频导入完毕

    # ---------------- 以上，导入视频数据完毕，接下来开始提取特征 -------------------------------------

    # Feature Extracting
    vid = vid.astype(np.float32)
    vid = 2.0 * (vid / 255.0) - 1.0   # 数据归一化
    if stream == 'rgb':
        vid = vid[:, :, :, ::-1]  # convert BGR to RGB  -- 这里是因为用cv2读取图片的通道格式为BGR,要转换为RGB，以方便卷积
    if stream == 'flow':
        channel = 2   # x channel & y channel
        vid = vid[:, :, :, 0:2]  # BG, not R
    # 每次处理100个segment，不足100另算
    # for j in range(num_hundred_segments + 1):
    #     if j == num_hundred_segments:
    #         extract_size = num_segments - num_hundred_segments * 100
    #     else:
    #         extract_size = 100
    for j in range(num_hundred_segments + 1):
        if j == num_hundred_segments:
            extract_size = num_segments - num_hundred_segments * 50
        else:
            extract_size = 50

        # tf.reset_default_graph()  # 新版本已被替换
        tf.compat.v1.reset_default_graph()
        feature_saver, feature_input, model_logits = uf.get_model(stream, extract_size)
        frame_inputs = np.zeros((extract_size, INPUT_VIDEO_FRAMES, IMAGE_SIZE, IMAGE_SIZE, channel))
        for k in range(extract_size):
            # frame_inputs[k] = vid[j * INPUT_VIDEO_FRAMES * 100 + k * INPUT_VIDEO_FRAMES:j * INPUT_VIDEO_FRAMES * 100 + (k + 1) * INPUT_VIDEO_FRAMES]
            frame_inputs[k] = vid[j * INPUT_VIDEO_FRAMES * 50 + k * INPUT_VIDEO_FRAMES:j * INPUT_VIDEO_FRAMES * 50 + (
                        k + 1) * INPUT_VIDEO_FRAMES]
        # X_feature[j * 100: j * 100 + extract_size] = uf.get_feature(frame_inputs, stream, extract_size, feature_saver, feature_input, model_logits)
        X_feature[j * 50: j * 50 + extract_size] = uf.get_feature(frame_inputs, stream, extract_size, feature_saver,
                                                                    feature_input, model_logits)

    # Save X_feature
    print('{:d} feature extracted'.format(i))
    npName = os.path.join(SAVE_PATHS[data][stream], '{:d}.npy'.format(i))
    np.save(npName, X_feature)  # 以numpy格式保存数据
    print('{:d} feature saved'.format(i))
