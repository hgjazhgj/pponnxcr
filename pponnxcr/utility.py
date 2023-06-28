import pkgutil

LANG={
    'en': {
        'det': 'en_PP-OCRv3_det_infer.onnx',
        'rec': 'en_PP-OCRv3_rec_infer.onnx',
        'cls': 'ch_ppocr_mobile_v2.0_cls_infer.onnx',
        #'dict': 'EN_symbol_dict.txt',
        'dict': 'en_dict.txt',
    },
    'zh': {
        'det': 'ch_PP-OCRv3_det_infer.onnx',
        'rec': 'ch_PP-OCRv3_rec_infer.onnx',
        'cls': 'ch_ppocr_mobile_v2.0_cls_infer.onnx',
        'dict': 'ppocr_keys_v1.txt',
    },
    'ja': {
        'det': 'Multilingual_PP-OCRv3_det_infer.onnx',
        'rec': 'japan_PP-OCRv3_rec_infer.onnx',
        'cls': 'ch_ppocr_mobile_v2.0_cls_infer.onnx',
        'dict': 'japan_dict.txt',
    },
}


def get_model_data(lang, step):
    return pkgutil.get_data(__name__, 'model/' + LANG[lang][step])


def get_character_dict(lang):
    return pkgutil.get_data(__name__, 'model/' + LANG[lang]['dict']).decode('utf-8').splitlines()


class OperatorGroup:
    def __init__(self, *ops):
        self.ops = ops

    def __call__(self, data):
        for op in self.ops:
            data = op(data)
        return data