import numpy as np
import cv2
import matplotlib.pyplot as plt

class Dataset:
    # resize_size: shape after resize
    def __init__(self, resize_size=256):
        self.data = {}
        self.SIZE = resize_size
        # Category and Attribute Prediction Benchmark
        path = "./DeepFashion/Category and Attribute Prediction Benchmark/"
        with open(path+"Anno/list_bbox.txt") as file:
            number = int(file.readline())
            file.readline()
            for _ in range(number):
                tmp = file.readline().split()
                self.data[path+tmp[0]] = {'bbox': [int(tmp[1]), int(tmp[2]), int(tmp[3]), int(tmp[4])], 'type': 0}
                
        cate_dict = {}
        with open(path+"Anno/list_category_cloth.txt") as file:
            number = int(file.readline())
            file.readline()
            for i in range(number):
                tmp = file.readline().split()
                cate_dict[i+1] = [tmp[0], int(tmp[1])]
                
        with open(path+"Anno/list_category_img.txt") as file:
            number = int(file.readline())
            file.readline()
            for _ in range(number):
                tmp = file.readline().split()
                self.data[path+tmp[0]]['type'] = cate_dict[int(tmp[1])][1]

        print(path + ": "+str(number))
        
        # Consumer-to-shop Clothes Retrieval Benchmark
        path = "./DeepFashion/Consumer-to-shop Clothes Retrieval Benchmark/"
        with open(path+"Anno/list_bbox_consumer2shop.txt") as file:
            number = int(file.readline())
            file.readline()
            for _ in range(number):
                tmp = file.readline().split()
                self.data[path+tmp[0]] = {'bbox': [int(tmp[3]), int(tmp[4]), int(tmp[5]), int(tmp[6])], 'type': tmp[1]}

        print(path + ": "+str(number))

        # Fashion Landmark Detection Benchmark
        path = "./DeepFashion/Fashion Landmark Detection Benchmark/"
        with open(path+"Anno/list_bbox.txt") as file:
            number = int(file.readline())
            file.readline()
            for _ in range(number):
                tmp = file.readline().split()
                self.data[path+tmp[0]] = {'bbox': [int(tmp[1]), int(tmp[2]), int(tmp[3]), int(tmp[4])], 'type': 0}

        with open(path+"Anno/list_joints.txt") as file:
            number = int(file.readline())
            file.readline()
            for _ in range(number):
                tmp = file.readline().split()
                self.data[path+tmp[0]]['type'] = int(tmp[1])        

        print(path + ": "+str(number))

        # In-shop Clothes Retrieval Benchmark
        path = "./DeepFashion/In-shop Clothes Retrieval Benchmark/"
        with open(path+"Anno/list_bbox_inshop.txt") as file:
            number = int(file.readline())
            file.readline()
            for _ in range(number):
                tmp = file.readline().split()
                self.data[path+tmp[0]] = {'bbox': [int(tmp[3]), int(tmp[4]), int(tmp[5]), int(tmp[6])], 'type': tmp[1]}
                
        print(path + ": "+str(number))

        self.data_size = len(self.data)


    def bbox_resize(self, bbox, origin, new):
        bb = [0,0,0,0]
        bb[0] = int(bbox[0]/origin[1]*new[1])
        bb[2] = int(bbox[2]/origin[1]*new[1])
        bb[1] = int(bbox[1]/origin[0]*new[0])
        bb[3] = int(bbox[3]/origin[0]*new[0])
        return bb

    # imgs: a list of images index. return a list with dict ('path': img_path(string), 'bbox': bbox(list), 'type': cloth_type(int), 'img': img(matrix))
    def get_data(self, imgs):
        key_list = list(np.array(list(self.data.keys()))[np.array(imgs)])
        result = [{'path': None, 'bbox': None, 'type': None, 'img': None} for i in range(len(imgs))]
        i = 0
        for k in key_list:
            img = cv2.imread(k)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
            result[i]['type'] = self.data[k]['type']
            result[i]['path'] = k
            result[i]['bbox'] = self.bbox_resize(self.data[k]['bbox'], img.shape, (self.SIZE, self.SIZE, 3))
            result[i]['img'] = cv2.resize(img,(256,256))
            i+=1
        return result


    # img_item: a list of images dict, came from 'get_data'; subplot_size: plot shape
    def show_data(self, img_item, subplot_size):
        i=1
        for item in img_item:
            img = item['img']
            cv2.rectangle(img, (item['bbox'][0], item['bbox'][1]), (item['bbox'][2], item['bbox'][3]), (0,255,0), 2)
            cv2.putText(img, str(item['type']), (item['bbox'][0]+5, item['bbox'][1]+20), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0) ,2)
            plt.subplot(subplot_size[0],subplot_size[1],i)
            i+=1
            plt.imshow(img)
            plt.axis("off")
        plt.show()


if __name__ == "__main__":
    dataset = Dataset()
    print(dataset.get_data([1,2,3]))
    dataset.show_data(dataset.get_data(np.random.randint(0,dataset.data_size,(25))), [5, 5])