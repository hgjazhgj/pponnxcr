import math
import time

import cv2
import numpy as np
import onnxruntime as ort

from ..utility import get_model_data, get_character_dict
from .rec_decoder import CTCLabelDecode


from ..log import get_logger
logger = get_logger('rec')


class TextRecognizer:
    def __init__(self, lang):
        self.rec_image_shape = [3, 48, 320]
        self.rec_batch_num = 6
        self.postprocess_op = CTCLabelDecode(character_dict=get_character_dict(lang))
        so = ort.SessionOptions()
        so.log_severity_level = 3
        sess = ort.InferenceSession(get_model_data(lang, 'rec'), so, providers=['CPUExecutionProvider'])
        self.predictor, self.input_tensor = sess, sess.get_inputs()[0]
        self.output_tensors = None

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        imgW = int(imgH * max_wh_ratio)
        resized_w = min(imgW, int(math.ceil(imgH * img.shape[1] / img.shape[0])))
        resized_image = cv2.resize(
            img,
            (resized_w, imgH),
            interpolation=cv2.INTER_CUBIC
        ).astype('float32').transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def __call__(self, img_list):
        img_num = len(img_list)
        # Sorting can speed up the recognition process
        indices = np.argsort([img.shape[1] / img.shape[0] for img in img_list])

        rec_res = [['', 0.0]] * img_num
        batch_num = self.rec_batch_num
        elapse = 0.
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            h, w = img_list[indices[end_img_no - 1]].shape[:2]
            norm_img_batch = np.concatenate([
                self.resize_norm_img(img_list[indices[ino]], w / h)[None]
                for ino in range(beg_img_no, end_img_no)
            ]).copy()
            starttime = time.time()
            input_dict = {self.input_tensor.name: norm_img_batch}

            outputs = self.predictor.run(self.output_tensors, input_dict)

            for rno, res in enumerate(self.postprocess_op(outputs[0])):
                rec_res[indices[beg_img_no + rno]] = res
            elapse += time.time() - starttime
        return rec_res, elapse
