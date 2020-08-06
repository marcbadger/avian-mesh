# keypoint order for optimization code is:
# [bill, left eye, right eye, neck, nape, left foot, right foot, tail tip,
#  left wrist, right wrist]
kpt_map = {
    0: 'bill tip',
    1: 'left eye',
    2: 'right eye',
    3: 'neck',
    4: 'nape',
    5: 'left wing tip',
    6: 'right wing tip',
    7: 'left foot',
    8: 'right foot',
    9: 'tail tip',
    10: 'left wrist',
    11: 'right wrist'
    }

# Camera
FOCAL_LENGTH = 2165
IMG_W = 1920
IMG_H = 1200

# Mean and standard deviation for normalizing input image
IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]

"""
Body_pose angle limit
we minius index by 1 because we exclude root pose as it is modeled as global orient
"""
max_lim = [0] * (24*3)
min_lim = [0] * (24*3)

# Lower body, Tail, Upper body, Chest
index = [1, 2, 3, 4]
for ind in index:
	i = ind - 1
	max_lim[i*3: (i+1)*3] = 0.5, 0.3, 0.3
	min_lim[i*3: (i+1)*3] = -0.5, -0.3, -0.3

# Hip
index = [5, 33-12]
for ind in index:
	i = ind - 1
	max_lim[i*3: (i+1)*3] = 0.3, 0., 0.
	min_lim[i*3: (i+1)*3] = -0.3, -0., -0.

# Thigh
index = [6, 34-12]
for ind in index:
	i = ind - 1
	max_lim[i*3: (i+1)*3] = 0.5, 0.2, 0.2
	min_lim[i*3: (i+1)*3] = -0.5, -0.2, -0.2

# Shin, Foot (left and right)
index = [7, 8, 35-12, 36-12]
value = [0.5, 0.4, 0.5, 0.4]
for (ind, val) in zip(index, value):
	i = ind - 1
	max_lim[i*3: (i+1)*3] = val, val, val
	min_lim[i*3: (i+1)*3] = -val, -val, -val

# Neck_1, Neck_2, Head
index = [25-12, 26-12, 27-12]
for ind in index:
	i = ind - 1
	max_lim[i*3: (i+1)*3] = 0.5, 0.5, 0.5
	min_lim[i*3: (i+1)*3] = -0.5, -0.5, -0.5



"""
Body bone length limit
"""
max_bone = [1] * (24)
min_bone = [1] * (24)

# Tail, Neck 2, Head
index = [2, 26-12, 27-12]
for ind in index:
	i = ind-1
	max_bone[i] = 1.5
	min_bone[i] = 0.5


# Foot, thigh
index = [8, 36-12, 6, 34-12]
for ind in index:
	i = ind-1
	max_bone[i] = 1.2
	min_bone[i] = 0.8

# 
index = [24-12, 32-12]
for ind in index:
	i = ind-1
	max_bone[i] = 1
	min_bone[i] = 0.5




