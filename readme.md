# pponnxcr - PaddlePaddle ONNXruntime OCR

OCR based on ONNX Runtime with PaddleOCR models  
Refractor by [@hgjazhgj](https://github.com/hgjazhgj/) for [ppocr-onnx](https://github.com/triwinds/ppocr-onnx) and implements:  

- Update `rec` model to v3  
- Remove all unnecessary components e.g. `draw-ocr`  
- Add En and Ja language support  

## Install

```bash
pip install pponnxcr
```

## Usage

```python
from pponnxcr import TextSystem
import cv2

ZH = TextSystem('zh')
JA = TextSystem('ja')
EN = TextSystem('en')

img = cv2.imread('test.png')

ZH.ocr_single_line(img)
ZH.ocr_lines([img, ...])
ZH.detect_and_ocr(img)
```

## License

[GNU AGPLv3](https://www.gnu.org/licenses/agpl-3.0.html)  

Which means that unless commercially licensed, any modification or use of this project in any way requires open source  

## Reference

- [ppocr-onnx](https://github.com/triwinds/ppocr-onnx)  
- [RapidOCR](https://github.com/RapidAI/RapidOCR)  
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)  
- [Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX)
