"""
    使用opencv的TV-L1算法提取光流
    参考：1、https://blog.csdn.net/qq_32799915/article/details/85704240 -- （主体代码）
          2、https://blog.csdn.net/xbinworld/article/details/50650319 --（这个主要讲基本原理）
          3、https://blog.csdn.net/weixin_41558411/article/details/89855290 -- （一些细节）
    细节：
        使用的是opencv-contrib-python包,如果装的是opencv-python，则需要将其卸载，然后重装opencv-contrib-python:
        pip3 uninstall opencv-python
        pip3 install opencv-contrib-python
"""
import os
import numpy as np
import cv2
from glob import glob
import argparse

from time import time

_IMAGE_SIZE = 224


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='train', type=str)
    argms = parser.parse_args()
    return argms


# Main : extract features
args = parse_args()
data = args.data

if data == 'train':
    data_path = r'/home/fzy/THUMOS14/train_frames/'
    save_path = "/home/huanbin/myproject/tpsn_feat_extraction/flow_frames/flow_train/"
elif data == 'test':
    data_path = r'/home/fzy/THUMOS14/test_frames/'
    save_path = "/home/huanbin/myproject/tpsn_feat_extraction/flow_frames/flow_test/"

video_list = os.listdir(data_path)
video_list = sorted(video_list)

if data == 'test':
    # I excluded two falsely annotated videos, 270, 1496, following the SSN paper (https://arxiv.org/pdf/1704.06228.pdf)
    video_list.remove('video_test_0000270')
    video_list.remove('video_test_0001496')
    video_list.remove('video_test_0001558')  # 删除最后一个,凑齐210

# huanbin add to write log txt
LOG_DIR = './'   # 当前目录下
LOG_FOUT = open(os.path.join(LOG_DIR, 'hb_get_'+str(data)+'_flow.txt'), 'w')

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


# 这里的video_path指向某一个视频的帧图片文件夹
def cal_for_frames(video_path):
    frames = glob(os.path.join(data_path + video_path, '*.jpg'))  # 获得该视频的所有图像帧的路径，此时图片无序
    frames.sort()  # 对图片路径排序，使得输入的图片有序

    flow = []
    prev = cv2.imread(frames[0])  # 初始，读取第一张图片
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)  # 转化为灰度图
    # 逐帧读取图像，并依次计算相邻两帧间的光流
    for i, frame_curr in enumerate(frames):
        curr = cv2.imread(frame_curr)
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        tmp_flow = compute_TVL1(prev, curr)  # 计算相邻两帧的光流
        flow.append(tmp_flow)
        prev = curr

    return flow


def compute_TVL1(prev, curr, bound=15):
    """Compute the TV-L1 optical flow."""
    # TVL1 = cv2.DualTVL1OpticalFlow_create()  # 注意这个类存在于包opencv-contrib-python中
    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = TVL1.calc(prev, curr, None)
    assert flow.dtype == np.float32

    flow = (flow + bound) * (255.0 / (2 * bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0

    return flow


def save_flow(video_flows, flow_path):
    # 分别保存x轴、y轴光流图
    for i, flow in enumerate(video_flows):
        cv2.imwrite(os.path.join(flow_path.format('u'), "{:06d}.jpg".format(i)),
                    flow[:, :, 0])
        cv2.imwrite(os.path.join(flow_path.format('v'), "{:06d}.jpg".format(i)),
                    flow[:, :, 1])


def extract_flow(video_path, flow_path):
    tic = time()
    log_string('start get flow from: '+str(data_path+video_path))
    flow = cal_for_frames(video_path)
    save_flow(flow, flow_path)
    log_string('complete and save to: ' + str(flow_path))
    toc = time()
    log_string('this video cost time: '+str( toc-tic)+ 'sec')
    return


if __name__ == '__main__':
    # video_paths = "/home/fzy/THUMOS14/train_frames/video_validation_0000990"   # 视频图像帧地址
    # flow_paths = "/home/huanbin/myproject/tpsn_feat_extraction/flow_frames/flow_train"  # 光流图像保存地址
    # video_lengths = 109  # en...这个好像没用
    total_tic = time()
    for i in range(len(video_list)):
        flow_save_path = save_path + video_list[i]
        if not os.path.exists(flow_save_path):
            os.makedirs(flow_save_path)
        extract_flow(video_list[i], flow_save_path)
    total_toc = time()
    log_string('total videos cost time: ' + str(total_toc-total_tic) + 'sec')




