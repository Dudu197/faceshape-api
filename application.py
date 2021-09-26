from flask import Flask
from flask import request
import json

from tensorflow import keras
from PIL import Image, ImageOps
import numpy as np

model = keras.models.load_model('model_128_sgd_reg.h5')
classes = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']

app = Flask(__name__)

def resize_image(img):
    old_size = img.size  # old_size[0] is in (width, height) format

    ratio = float(128) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    img = img.resize((128, 128), Image.ANTIALIAS)

    new_im = Image.new("RGB", (128, 128))
    new_im.paste(img, ((128 - new_size[0]) // 2,
                      (128 - new_size[1]) // 2))
    return new_im

@app.route('/', methods=['POST'])
def example():
   img = Image.open(request.files['file'].stream)
   img = resize_image(img)
   img = ImageOps.grayscale(img)
   img_arr = np.array(img).reshape(-1, 128, 128, 1)
   predict = model.predict(img_arr)[0]
   max_index = predict.argmax()
   result = {
      'class': classes[max_index],
      'precision': float(predict[max_index])
   }
   return json.dumps(result)

if __name__ == '__main__':
    app.run()
