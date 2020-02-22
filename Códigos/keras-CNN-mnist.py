from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
import numpy as np
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

from keras.optimizers import SGD
#K.set_image_dim_ordering('th')
K.set_image_data_format('channels_first')

# Construimos nossos subconjuntos de treinamento e teste.
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Como estamos trabalhando em escala de cores cinza podemos
# definir a dimensao do pixel como sendo 1.
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# Normalizamos nossos dados de acordo com variacao da
# escala de cinza.
X_train = X_train / 255
X_test = X_test / 255

# Aplicamos a solucao de one-hot-encoding para
# classificacao multiclasses.
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Numero de tipos de digitos encontrados no MNIST.
num_classes = y_test.shape[1]

def deeper_cnn_model():
    model = Sequential()

    # A Convolution2D sera a nossa camada de entrada. Podemos observar que ela possui 
    # 30 mapas de features com tamanho de 5 × 5 e 'relu' como funcao de ativacao. 
    model.add(Conv2D(60, (5, 5), input_shape=(1, 28, 28), activation='relu'))

    # A camada MaxPooling2D sera nossa segunda camada onde teremos um amostragem de 
    # dimensoes 2 × 2.
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Uma nova camada convolucional com 15 mapas de features com dimensoes de 3 × 3 
    # e 'relu' como funcao de ativacao. 
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    # Dropout com probabilidade de 30%
    model.add(Dropout(0.3))

    # Flatten preparando os dados para a camada fully connected. 
    model.add(Flatten())

    # Camada fully connected de 128 neuronios.
    model.add(Dense(512, activation='relu'))

    # A camada de saida possui o numero de neuronios compativel com o 
    # numero de classes a serem classificadas, com uma funcao de ativacao
    # do tipo 'softmax'.
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

model = deeper_cnn_model()

# O metodo summary revela quais sao as camadas
# que formam o modelo, seus formatos e o numero
# de parametros envolvidos em cada etapa.
model.summary()

# Processo de treinamento do modelo. 
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=100)

# Avaliacao da performance do nosso primeiro modelo.
scores = model.evaluate(X_test, y_test, verbose=0)
print("Erro de: %.2f%%" % (100-scores[1]*100))