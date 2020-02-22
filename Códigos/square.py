from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
import numpy as np
from keras.models import load_model
from keras.layers.convolutional import Conv2D
from keras.layers import Reshape
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import callbacks

K.set_image_data_format('channels_last')

model = Sequential()
model.add(Reshape((1,100,100), input_shape=(100, 100, 1)))
model.add(Conv2D(20, (5, 5), input_shape=(1, 100, 100), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Conv2D(10, (3, 3), activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

train_datagen = ImageDataGenerator(rescale=1./255,
		horizontal_flip=True)

X_train = train_datagen.flow_from_directory(
        'image',
        target_size=(100, 100),
        batch_size=10,
        color_mode='grayscale')

filepath = 'weights.hdf5'

checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, mode='max')

callbacks_list = [checkpoint]

model.fit_generator(
    X_train,
    steps_per_epoch=20,
    epochs=10,
    callbacks=callbacks_list)

# model = load_model('weights.hdf5')

img = image.load_img('teste/4.png', target_size=(100, 100), color_mode='grayscale')
y = image.img_to_array(img)
y = np.expand_dims(y, axis=0)

classes = model.predict_classes(y)
print(classes[0])

if(classes[0]==0): print("QUADRADO!")
else: print("TRIANGULO!")