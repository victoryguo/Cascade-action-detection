import numpy as np
np.set_printoptions(threshold=np.inf)

class_txt = './annotation_val/detclasslist.txt'

class_dict = {}
class_list = []
with open(class_txt, 'r') as fr:
    lines = fr.readlines()
    for i in range(len(lines)):
        line = lines[i]
        index = line.split(' ')[0]
        class_name = line.split(' ')[-1].strip('\n')
        class_list.append(class_name)
        class_dict[int(index)-1] = class_name
# print('class_dict: ', class_dict)
# print('class_list: ', class_list)

def delete_same_item(inp_list):
    ret_list = []
    for i in inp_list:
        if not i in ret_list:
            ret_list.append(i)
    return ret_list

class_file_path = ['./annotation_val/'+ i + '_val.txt' for i in class_list]
# print(class_file_path)
video_list = []
for i in range(len(class_list)):
    f = open(class_file_path[i], 'r')
    lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i]
        video_name = line.split(' ')[0]
        video_list.append(video_name)
f.close()
video_list = sorted(delete_same_item(video_list))
# print('video_list: ', len(video_list),'\n', video_list)


def _check_video_in_class(class_list, class_file_path, video_name, class_name):
    flag = 0
    for i in range(len(class_list)):
        if class_list[i] == class_name:
            f = open(class_file_path[i], 'r')
            lines = f.readlines()
            for i in range(len(lines)):
                line = lines[i]
                vid_name = line.split(' ')[0]
                if vid_name == video_name:
                    flag = 1
    return flag


train_labels = np.zeros([200, 20])
for i in range(len(video_list)):
    for j in range(len(class_list)):
        if _check_video_in_class(class_list, class_file_path, video_list[i], class_list[j]):
            train_labels[i][j] = 1

print('train_labels : \n', train_labels)

np.save('train_labels.npy', train_labels)

# total_one = 0
# for i in range(train_labels.shape[0]):
#     for j in range(train_labels.shape[1]):
#         if train_labels[i][j] == 1:
#             total_one += 1
# print('total_one: ', total_one)  # 227



