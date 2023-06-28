import copy
import math
import time

import cv2
import numpy as np
import onnxruntime as ort
from ..utility import get_model_data
from .postprocess import ClsPostProcess

from ..log import get_logger
logger = get_logger('cls')


class TextClassifier:
    def __init__(self, lang, label_list=('0', '180'), cls_batch_num=6, cls_thresh=0.9):
        self.cls_image_shape = [3, 48, 192]
        self.cls_batch_num = cls_batch_num
        self.cls_thresh = cls_thresh
        self.postprocess_op = ClsPostProcess(label_list=label_list)
        self.output_tensors = None
        sess = ort.InferenceSession(get_model_data(lang, 'cls'), providers=['CPUExecutionProvider'])
        self.predictor, self.input_tensor = sess, sess.get_inputs()[0]

    def resize_norm_img(self, img):
        imgC, imgH, imgW = self.cls_image_shape
        resized_w = min(imgW, int(math.ceil(imgH * img.shape[1]/img.shape[0])))
        resized_image = cv2.resize(img, (resized_w, imgH)).astype('float32')
        if self.cls_image_shape[0] == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[None]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def __call__(self, img_list):
        img_list = copy.deepcopy(img_list)
        img_num = len(img_list)
        # Sorting can speed up the cls process
        indices = np.argsort([img.shape[1] / img.shape[0] for img in img_list])

        cls_res = [['', 0.0]] * img_num
        batch_num = self.cls_batch_num
        elapse = 0.
        for beg_img_no in range(0, img_num, batch_num):
            norm_img_batch = np.concatenate([
                self.resize_norm_img(img_list[indices[ino]])[None]
                for ino in range(beg_img_no, min(img_num, beg_img_no + batch_num))
            ]).copy()
            starttime = time.time()
            input_dict = {self.input_tensor.name: norm_img_batch}

            outputs = self.predictor.run(self.output_tensors, input_dict)

            cls_result = self.postprocess_op(outputs[0])
            elapse += time.time() - starttime
            for rno in range(len(cls_result)):
                label, score = cls_result[rno]
                cls_res[indices[beg_img_no + rno]] = [label, score]
                if '180' in label and score > self.cls_thresh:
                    img_list[indices[beg_img_no + rno]] = cv2.rotate(
                        img_list[indices[beg_img_no + rno]],
                        cv2.ROTATE_180
                    )
        return img_list, cls_res, elapse
