import matplotlib.pyplot as plt
import numpy as np

# 生成画布
ax = plt.figure(figsize=(15, 8), dpi=80)

tIoU = [0.1, 0.2, 0.3, 0.4, 0.5]
x = range(len(tIoU))

# thumos14 different number of erased branches results
# y0 = [0.52, 0.447, 0.355, 0.258, 0.169]  # STPN result
y1 = [0.579, 0.508, 0.414, 0.301, 0.206]  # one erased branch result
y2 = [0.521, 0.445, 0.359, 0.271, 0.178]  # two erased branch result
y3 = [0.534, 0.469, 0.381, 0.278, 0.175]  # W/O erased branch result

# activitynet1.2 different number of erased branches results
# y1 = [0.531, 0.481, 0.436, 0.39, 0.345]  # one erased branch result
# y2 = [0.524, 0.477, 0.433, 0.387, 0.342] # two erased branch result
# y3 = [0.526, 0.474, 0.429, 0.383, 0.338] # W/O erased branch result

# plt.bar([i - 0.25 for i in x], y0, width=0.25, color=['g'])
plt.bar([i - 0.25 for i in x], y3, width=0.25, color=['b'], label='W/O erased branch')
plt.bar(x, y1, width=0.25, color=['r'], label='one erased branch')
plt.bar([i + 0.25 for i in x], y2, width=0.25, color=['y'], label='two erased branch')

# activitynet1.2 different stream results
# y1 = [0.531, 0.481, 0.436, 0.39, 0.345]  # fusion
# y2 = [0.499, 0.454, 0.412, 0.37, 0.326]  # rgb only
# y3 = [0.453, 0.415, 0.379, 0.342, 0.302] # flow only

# thumos14 different stream results
# y1 = [0.579, 0.508, 0.414, 0.301, 0.206] # fusion
# y2 = [0.365, 0.297, 0.213, 0.142, 0.087] # rgb only
# y3 = [0.46, 0.4, 0.32, 0.238, 0.157] # flow only
#
# plt.bar([i - 0.25 for i in x], y1, width=0.25, color=['b'], label='RGB + FLOW')
# plt.bar(x, y2, width=0.25, color=['r'], label='RGB')
# plt.bar([i + 0.25 for i in x], y3, width=0.25, color=['y'], label='FLOW')


plt.tick_params(labelsize=20)


plt.legend(fontsize=20)
plt.xticks(x, tIoU)
plt.xlabel(r'IoU', fontsize=20)
plt.ylabel(r'Mean Average Precision', fontsize=20)

plt.show()
