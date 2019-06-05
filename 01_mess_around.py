from urllib.request import urlopen
from io import BytesIO

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt


############### draw plot ##################
model = ResNet50(weights='imagenet')

img_url = 'http://media-cdn.tripadvisor.com/media/photo-s/07/58/91/7c/elephant-nature-park.jpg'
img_fp = BytesIO(urlopen(img_url).read())
img = image.load_img(img_fp, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]


############### draw plot ##################
X = np.linspace(0, 2 * np.pi, 10)
plt.plot(X, np.sin(X), '-o')
plt.title('Sine curve')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'sin($\alpha$)')

plt.show()