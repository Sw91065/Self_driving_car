print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import h5py
print(h5py.__version__)

from utils import *
from sklearn.model_selection import train_test_split
from keras.api.models import save_model

##Step 1

path='mydata'
data=importdatainfo(path)

#Step 2
data=balanceData(data, display=False)

#Step 3
imagesPath, steerings = loadData(path,data)

#Step 4
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2,random_state=10)
print('Total Training Images: ',len(xTrain))
print('Total Validation Images: ',len(xVal))

#step 8
model = createModel()
model.summary()

#step 9

history = model.fit(batchGen(xTrain, yTrain, 100, 1),
                                  steps_per_epoch=20,
                                  epochs=2,
                                  validation_data=batchGen(xVal, yVal, 10, 0),
                                  validation_steps=20)

#step 10
# model.save('model.keras')
save_model(model, 'model.h5')
print('Model Saved')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()