class ClsPostProcess:
    """ Convert between text-label and text-index """

    def __init__(self, label_list):
        self.label_list = label_list

    def __call__(self, preds, label=None):
        decode_out = [(self.label_list[idx], preds[i, idx])
                      for i, idx in enumerate(preds.argmax(axis=1))]
        if label is None:
            return decode_out
        label = [(self.label_list[idx], 1.) for idx in label]
        return decode_out, label
