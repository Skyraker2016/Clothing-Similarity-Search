import numpy as np
import cv2
import matplotlib.pyplot as plt
data = {}
SIZE = 256
path_CAPB = "./DeepFashion/Category and Attribute Prediction Benchmark/"
with open(path_CAPB+"Anno/list_bbox.txt") as file:
    number = int(file.readline())
    file.readline()
    for _ in range(number):
        tmp = file.readline().split()
#         if (path_CAPB+tmp[0] in data.keys()):
#             print(tmp[0])
#             data[path_CAPB+tmp[0]].append({'bbox': [int(tmp[1]), int(tmp[2]), int(tmp[3]), int(tmp[4])], 'category': 0, 'type': 0})
#         else:
#             data[path_CAPB+tmp[0]] = [{'bbox': [int(tmp[1]), int(tmp[2]), int(tmp[3]), int(tmp[4])], 'category': 0, 'type': 0}]
        data[path_CAPB+tmp[0]] = {'bbox': [int(tmp[1]), int(tmp[2]), int(tmp[3]), int(tmp[4])], 'type': 0}
        
cate_dict = {}
with open(path_CAPB+"Anno/list_category_cloth.txt") as file:
    number = int(file.readline())
    file.readline()
    for i in range(number):
        tmp = file.readline().split()
        cate_dict[i+1] = [tmp[0], int(tmp[1])]
        
with open(path_CAPB+"Anno/list_category_img.txt") as file:
    number = int(file.readline())
    file.readline()
    for _ in range(number):
        tmp = file.readline().split()
        data[path_CAPB+tmp[0]]['type'] = cate_dict[int(tmp[1])][1]

print(path_CAPB + ": "+str(number))

path_CAPB = "./DeepFashion/Consumer-to-shop Clothes Retrieval Benchmark/"
with open(path_CAPB+"Anno/list_bbox_consumer2shop.txt") as file:
    number = int(file.readline())
    file.readline()
    for _ in range(number):
        tmp = file.readline().split()
        data[path_CAPB+tmp[0]] = {'bbox': [int(tmp[3]), int(tmp[4]), int(tmp[5]), int(tmp[6])], 'type': tmp[1]}

print(path_CAPB + ": "+str(number))

path_CAPB = "./DeepFashion/Fashion Landmark Detection Benchmark/"
with open(path_CAPB+"Anno/list_bbox.txt") as file:
    number = int(file.readline())
    file.readline()
    for _ in range(number):
        tmp = file.readline().split()
        data[path_CAPB+tmp[0]] = {'bbox': [int(tmp[1]), int(tmp[2]), int(tmp[3]), int(tmp[4])], 'type': 0}

with open(path_CAPB+"Anno/list_joints.txt") as file:
    number = int(file.readline())
    file.readline()
    for _ in range(number):
        tmp = file.readline().split()
        data[path_CAPB+tmp[0]]['type'] = int(tmp[1])        

print(path_CAPB + ": "+str(number))

path_CAPB = "./DeepFashion/In-shop Clothes Retrieval Benchmark/"
with open(path_CAPB+"Anno/list_bbox_inshop.txt") as file:
    number = int(file.readline())
    file.readline()
    for _ in range(number):
        tmp = file.readline().split()
        data[path_CAPB+tmp[0]] = {'bbox': [int(tmp[3]), int(tmp[4]), int(tmp[5]), int(tmp[6])], 'type': tmp[1]}
        
print(path_CAPB + ": "+str(number))

def bbox_resize(bbox, origin, new):
    bb = [0,0,0,0]
    bb[0] = int(bbox[0]/origin[1]*new[1])
    bb[2] = int(bbox[2]/origin[1]*new[1])
    bb[1] = int(bbox[1]/origin[0]*new[0])
    bb[3] = int(bbox[3]/origin[0]*new[0])
    return bb

def get_data(imgs):
    key_list = list(np.array(list(data.keys()))[np.array(imgs)])
    result = {}
    for k in key_list:
        img = cv2.imread(k)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        result[k] = data[k]
        result[k]['bbox'] = bbox_resize(result[k]['bbox'], img.shape, (SIZE, SIZE, 3))
        result[k]['img'] = cv2.resize(img,(256,256))
    return result



def show_data(img_item, subplot_size, fs=[10,10]):
    test = list(img_item.keys())
    plt.figure(figsize=fs)
    i=1
    for t in test:
        img = data[t]['img']
        cv2.rectangle(img, (data[t]['bbox'][0], data[t]['bbox'][1]), (data[t]['bbox'][2], data[t]['bbox'][3]), (0,255,0), 2)
        cv2.putText(img, str(data[t]['type']), (data[t]['bbox'][0]+5, data[t]['bbox'][1]+20),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0) ,2)
        plt.subplot(subplot_size[0],subplot_size[1],i)
        i+=1
        plt.imshow(img)
        plt.axis("off")
    plt.show()
