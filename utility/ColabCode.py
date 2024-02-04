import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
import keras
from keras.callbacks import ReduceLROnPlateau
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras.callbacks import ModelCheckpoint

"""
You can use this code to train your own model in google colab
"""

print("GPU Available: ", tf.config.list_physical_devices('GPU'))

image_width = 224
image_height = 224
color = 1

epc = 30
bs = 2

def model(input_shape=(224, 224, 1), num_classes=5):
    model = Sequential()

    model.add(Conv2D(32, input_shape = input_shape, padding="same", kernel_size=(3, 3), activation="relu"))
    model.add( MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax", kernel_regularizer='l1_l2'))
    return model

data = np.load(r"write your image arrays path here")
labels = np.load(r"write your image labels path here")

labelEn = LabelEncoder()
labels = labelEn.fit_transform(labels)
labels = to_categorical(labels)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=.10, shuffle=True)

x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)

x_test_mean = np.mean(x_test)
x_test_std = np.std(x_test)

x_train = (x_train - x_train_mean)/x_train_std
x_test = (x_test - x_test_mean)/x_test_std

x_train, x_validate, y_train, y_validate = train_test_split(
    x_train, y_train, test_size = .10, shuffle = True,random_state=42)

model = model(input_shape=(image_width, image_height, color), num_classes=5)
model.summary()

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1,  factor=0.5, min_lr=0.00001)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

checkpoint = ModelCheckpoint("write here where you want to save the model checkpoint path", monitor='val_accuracy',
                             verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit(x_train,y_train, batch_size=bs,
                              epochs = epc, validation_data = (x_validate,y_validate),
                              verbose = 1, callbacks=[learning_rate_reduction, callbacks_list])

checkpointPath = "your checkpoint path" + "weights.best.hdf5"
model.load_weights(checkpointPath)

history.history.keys()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

Y_pred = model.predict(x_test)
Y_pred_classes = np.argmax(Y_pred,axis = 1)
Y_true = np.argmax(y_test,axis = 1)
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
plot_confusion_matrix(confusion_mtx, classes = range(3))

report = classification_report(Y_true, Y_pred_classes)
print(report)

os.chdir(r"write here where you want to save the trained model path")
model.save('model1.h5')

preds = model.predict(x_test)
y_pred = np.zeros_like(preds)
y_pred[np.arange(len(preds)), preds.argmax(1)] = 1
classes = ['A','L','R','T','W']
confusionMatrix = np.zeros((len(classes),len(classes)))

for i in range(len(y_test)):

  if np.array_equal(y_pred[i],y_test[i]):
    index = np.argmax(y_test[i])
    confusionMatrix[index,index] += 1

  else:

    index1 = np.argmax(y_test[i])
    index2 = np.argmax(y_pred[i])
    confusionMatrix[index1,index2] += 1

print("CLASSIFICATION REPORT")
print(classification_report(y_test, y_pred))
print("Accuracy Score : ")
print(accuracy_score(y_test, y_pred))
cm = accuracy_score(y_test, y_pred)
sns.set(font_scale=0.8)
sns.heatmap(confusionMatrix,annot=True, linewidths=1.0, cbar=False)
print("--------------------------------------------------")
print("Confusion Matrix : ")
print(confusionMatrix)
print("--------------------------------------------------")

