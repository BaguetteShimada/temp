import os
import json
import cv2
import numpy as np
from skimage.feature import greycomatrix


class TraitExtra:
    def bgrTohsv(self, bgrimg):  # bgr图像转hsv图像
        hsvimg = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2HSV)
        return hsvimg

    def mean(self, hsvimg):  # 计算hsv图像一阶矩
        h, s, v = cv2.split(hsvimg)
        h_mean = h.mean()
        s_mean = h.mean()
        v_mean = v.mean()
        return h_mean, s_mean, v_mean

    def std(self, hsvimg):  # 计算hsv图像二阶矩
        h, s, v = cv2.split(hsvimg)
        h_std = h.std()
        s_std = s.std()
        v_std = v.std()
        return h_std, s_std, v_std

    def bgrTogray(self, bgrimg, gray_level):  # bgr图转灰度图
        gray = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2GRAY)
        gray = self.gray_level_descend(gray, gray_level)
        return gray

    def gray_level_descend(self, grayimg, target_level):  # 降低灰度等级，减少运算量
        h, w = grayimg.shape
        new_img = np.zeros((h, w), int)
        for i in range(h):
            for j in range(w):
                new_img[i][j] = grayimg[i][j] % target_level
        return new_img

    def cal_glcm(self, grayimg, gray_level):  # 计算灰度共生矩阵
        glcm = greycomatrix(grayimg, [1], [0], levels=gray_level)
        return glcm[:, :, 0, 0]

    def cal_glcm_energy(self, glcm):  # 计算灰度共生矩阵能量
        h, w = glcm.shape
        energy = 0
        for i in range(h):
            for j in range(w):
                energy += glcm[i][j] ** 2
        return energy

    def cal_glcm_ent(self, glcm):  # 计算灰度共生矩阵熵
        h, w = glcm.shape
        ent = 0
        for i in range(h):
            for j in range(w):
                if glcm[i][j] != 0:
                    ent += glcm[i][j] * np.log2(glcm[i][j])
        ent *= -1
        return ent

    def cal_glcm_contrast(self, glcm):  # 计算灰度共生矩阵对比度
        h, w = glcm.shape
        cont = 0
        for i in range(h):
            for j in range(w):
                if glcm[i][j] != 0:
                    cont += (i - j) ** 2 * glcm[i][j]
        return cont

    def cal_glcm_homogene(self, glcm):  # 计算灰度共生矩阵同质性
        h, w = glcm.shape
        homo = 0
        for i in range(h):
            for j in range(w):
                if glcm[i][j] != 0:
                    homo += glcm[i][j] / (1 + (i - j) ** 2)
        return homo

    def prepare_data(self):
        try:  # 尝试读取记录数据
            with open(""
                      "", encoding="utf8") as f:
                train = json.load(f)
            return train
        except FileNotFoundError:
            pass
        train = {"imgpath": [], "target": [], "data": []}
        for root, dirs, files in os.walk("./data/airplane"):
            for file in files:
                train["imgpath"].append(os.path.join(root, file))
                train["target"].append(0)
        for root, dirs, files in os.walk("./data/forest"):
            for file in files:
                train["imgpath"].append(os.path.join(root, file))
                train["target"].append(1)
        for imgpath in train["imgpath"]:
            image = cv2.imread(imgpath)
            hsvimg = self.bgrTohsv(image)
            mean = self.mean(hsvimg)
            std = self.std(hsvimg)
            feature = list(mean + std)
            gray = self.bgrTogray(image, 32)
            glcm = self.cal_glcm(gray, 32)
            # feature.append(self.cal_glcm_ent(glcm))#熵
            # feature.append(self.cal_glcm_energy(glcm))#能量
            feature.append(self.cal_glcm_contrast(glcm))  # 对比度
            feature.append(self.cal_glcm_homogene(glcm))  # 均质性
            for i in range(len(feature)):
                feature[i] = float(feature[i])
            train["data"].append(feature)
            print(1, end=" ")
        with open("./data.json", "w", encoding="utf8") as f:  # 记录数据
            json.dump(train, f, indent=3)
        return train


if __name__ == '__main__':
    types = {"airplane": 0, "forest": 1}
    te = TraitExtra()
    data = te.prepare_data()
