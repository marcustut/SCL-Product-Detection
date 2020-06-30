import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import cv2

tf.debugging.set_log_device_placement(True)
train_data = pd.read_csv('./dataset/train.csv')
test_data = pd.read_csv('./dataset/test.csv')
train_DATADIR = "./dataset/train/train"
test_DATADIR = "./dataset/test/test"
train_images = []
test_images = []
train_labels = np.array(train_data.category.tolist())
test_labels = np.array(test_data.category.tolist())
class_names = ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41']
img_size = 224
batch_size = None
# print(class_names)
idx = 0
idx2 = 0

with tf.device('/job:localhost/replica:0/task:0/device:GPU:0'):
  if os.path.getsize('./dataset/train_images_bitmap.npy') == 0:
    for x in train_data.category.tolist():
        if x < 10:
            x = "0" + str(x)
            path = os.path.join(train_DATADIR,x)
        else:
            x = str(x)
            path = os.path.join(train_DATADIR,x)
        img_array = cv2.imread(os.path.join(path,str(train_data.filename[idx])), cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array,(img_size,img_size))
        train_images.append(new_array) 
        idx += 1
        print(f'{idx}/105392 - {(idx/105392)*100:.2f}%')
    narray = np.array(train_images)
    np.save('./dataset/train_images_bitmap.npy', narray)
    #flattenarray = narray.flatten(order='C')
    #flattenarray.tofile('train_images_bitmap.txt')
  else:
    print('The bitmap file is found and continue to Tensorflow now.')
    train_images = np.load('train_images_bitmap.npy')

_continue = input("Do you want to continue?")

if _continue.strip().lower() == 'n' or _continue.strip().lower() == 'no':
  sys.exit()

  # for img in os.listdir(path):
  #     img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
  #     new_array = cv2.resize(img_array,(img_size,img_size))
  #     train_images.append(new_array)
  #     # plt.imshow(img_array,cmap="gray")
  #     # plt.title(x)
  #     # plt.xlabel(train_data.filename[idx])
  #     # plt.show()   
  #     idx += 1
  #     print(f'{idx}/105398 - {idx/105398:.2f}%')


  train_images = np.array(train_images)
  # np.savetxt("trainimg_bitmap.txt",train_images,fmt="%s")
  train_images = train_images / 255.0

  # test_images = test_images / 255.0

  # plt.figure()
  # plt.imshow(train_images[0])
  # plt.colorbar()
  # plt.grid(False)
  # plt.show()

  # plt.figure(figsize=(10,10))
  # for i in range(25):
  #     plt.subplot(5,5,i+1)
  #     plt.xticks([])
  #     plt.yticks([])
  #     plt.grid(False)
  #     plt.imshow(train_images[i], cmap=plt.cm.binary)
  #     plt.xlabel(class_names[train_labels[i]])
  # plt.show()

model = keras.Sequential([
  # Input layer (Flatten into 28*28 pixels of 784 neurons)
    keras.layers.Flatten(input_shape=(100, 100)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(42)
])

#Loss smaller the better, Accuracy higher the better
model.compile(optimizer='SGD',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# Epoch 602 is closest
# epoch is a complete learning pass through the entire dataset, run how many times for the whole training set, x * 60k(which is the training set)
model.fit(train_images, train_labels, batch_size=batch_size,epochs=2000)
'''
# test the test images in this model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
# Test accuracy means the percentage of the images that are correctly classified
print('\nTest accuracy:', test_acc)

# Softmax normalizes output into probability distribution, not the large numbers
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

print(np.argmax(predictions[0]))

print(test_labels[0])

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
# num_rows = 5
# num_cols = 3
# num_images = num_rows*num_cols
# plt.figure(figsize=(2*2*num_cols, 2*num_rows))
# for i in range(num_images):
#   plt.subplot(num_rows, 2*num_cols, 2*i+1)
#   plot_image(i, predictions[i], test_labels, test_images)
#   plt.subplot(num_rows, 2*num_cols, 2*i+2)
#   plot_value_array(i, predictions[i], test_labels)
# plt.tight_layout()
# plt.show()

# Grab an image from the test dataset.
img = test_images[1]

print(img.shape)

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)

predictions_single = probability_model.predict(img)

print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

np.argmax(predictions_single[0])

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()
'''
loss_history = model.history["loss"]
np_loss_history = np.array(loss_history)
np.savetxt("lossHistory.txt", np_loss_history, delimiter=",")
model.save("shopeetrained20kE.h5")
