# Whale and Dolphin Identification Competition
> Kaggle competition (Happywhales - Whale and Dolphin Identification)

**Contents**

 * [Overview](#overview)
 * [Data Exploration](#data-exploration)
 * [Data Preperation](#data-preperation)
 * [EfficientNetB0](#efficientnetb0)
 * [Experiments](#experiments)
## Overview

For the purpose of showcasing my workflow, this readme will outline the process of going through the [Whale and Dolphin Identification](https://www.kaggle.com/c/happy-whale-and-dolphin/overview) Kaggle competition. For more overview information regarding this competition please visit its overview webpage.

The data in this competition contains images of over **15,000** unique individual marine mammals from **30** different species collected from **28** different research organizations. The individuals have been manually identified and given an `individual_id`. The task of this competition is to correctly identify these individuals in images. In modelling terms, this `individual_id` value is the target variable for prediction. 

|**Files**   | **Description**|
|--------|---------|
|train_images/| A folder containing the training images.|
|train.csv| Provides the `species` and the `individual_id` for each of the training images.|
|test_images/| A folder containing the test images; these are the images which the task of this competition is centered on. The task is to predict `individual_id` and should be predicted as `new_individual`.|
|sample_submission.csv| A sample submission file in the correct format.|

## Data Exploration

The modules used to explore this competition's dataset are imported as follows:

```python
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
```
We can then obtain the files and folders of the upper level competition dataset folder:
```python
print(f"Competition Files and Folders: {os.listdir('/kaggle/input/happy-whale-and-dolphin')}")
```
```output
Competition Files and Folders: ['sample_submission.csv', 'train_images', 'train.csv', 'test_images']
```
Next we can load the appropriate files to begin exploring:
```python
train_df = pd.read_csv('/kaggle/input/happy-whale-and-dolphin/train.csv')
samp_submission_df = pd.read_csv('/kaggle/input/happy-whale-and-dolphin/sample_submission.csv')
```
The `train_df` variable represents the **train.csv** data in a DataFrame format which contains the name of the training images: `image`, species name: `species`, and the `individual_id`.

Looking at `train_df` we can see:
```python
train_df.head()
```
| |`image`|`species`|`individual_id`|
|-|-------|---------|---------------|
|0| xxxx1.jpg| species_name1| xxxxid1|
|1| xxxx2.jpg| species_name2| xxxxid2|
|2| xxxx3.jpg| species_name3| xxxxid3|
|3| xxxx4.jpg| species_name4| xxxxid4|
|4| xxxx5.jpg| species_name5| xxxxid5|

Please visit my [competition kernel](https://www.kaggle.com/umbertofasci/happy-whales-and-dolphins-starter) to view this data more accurately.
You can also visit the [rendered version](https://github.com/UmbertoFasci/Whale_Dolphin_Identification_Competition/blob/main/happy-whales-and-dolphins-starter.ipynb) on github.

After exploring this data thoroughly we can find some spelling errors within the `species` column. To fix this we can simply relate these incorrect names to the correct ones and save it to the original dataframe:
```python
train_df.loc[train_df.species == 'kiler_whale', 'species'] = 'killer_whale'
train_df.loc[train_df.species == 'bottlenose_dolpin', 'species'] = 'bottlenose_dolphin'
```
We can now look at the `samp_submission_df` representing the format in which the competition file should be submitted:
```python
samp_submission_df.head()
```
| |`image`|`predictions`|
|-|-------|-------------|
|0| xxxx1.jpg| xxxxid1 xxxxid2 xxxxid3 xxxid4|
|1| xxxx2.jpg| xxxxid5 xxxxid6 xxxxid7 xxxid8|

We can now count the number of unique images, species, and individual IDs within `train_df` using the `nunique()` method which returns an integer value of the number of unique values in a column:
```python
print(f"Images in train index file: {train_df.image.nunique()}")
print(f"Species in train index file: {train_df.species.nunique()}")
print(f"Individual IDs in train index file: {train_df.individual_id.nunique()}")
```
We then determine the number of images within the test and train images folders:
```python
print(f"Images in train images folder: {len(os.listdir('/kaggle/input/happy-whale-and-dolphin/train_images'))}")
print(f"Images in test images folder: {len(os.listdir('/kaggle/input/happy-whale-and-dolphin/test_images'))}")
```
```output
Images in train index file: 51033 
Species in train index file: 28
Individual IDs in train index file: 15587
Images in train images folder: 51033
Images in test images folder: 27956
```
Let's look at the species' images frequency within `train_df`:
```python
spec_freq = train_df["species"].value_counts()
df = pd.DataFrame({'Species': spec_freq.index,
                   'Images': spec_freq.values})
plt.figure(figsize = (12, 6))
plt.title('Distribution of Species Images - Train Dataset')
sns.set_color_codes("deep")
s = sns.barplot(x = "Species", y="Images", data=df)
s.set_xticklabels(s.get_xticklabels(), rotation=90)
locs, labels = plt.xticks()
plt.show()
```
![Distribution of Species Images - Training Set](https://github.com/UmbertoFasci/Whale_Dolphin_Identification_Competition/blob/main/Distribution_of_Species_imgs.png)

**The `Bottlenose Dolphin` species has the most present images while the `Beluga Whale` and `Humpback Whale` come in as the second heighest present group.**

Now we visualize the individual IDs associated with each species:
```python
id_freq = train_df.groupby(["species"])["individual_id"].nunique()
df = pd.DataFrame({'Species': id_freq.index,
                   'Unique ID Count': id_freq.values
                  })
df = df.sort_values(['Unique ID Count'], ascending=False)
plt.figure(figsize = (12,6))
plt.title('Distribution of Species Individual IDs - train dataset')
sns.set_color_codes("deep")
s = sns.barplot(x = 'Species', y="Unique ID Count", data=df)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
locs, labels = plt.xticks()
plt.show()
```
![Distribution of Species IDs - Training Set](https://github.com/UmbertoFasci/Whale_Dolphin_Identification_Competition/blob/main/Distribution_of_Species_ID.png)

**The `Dusky Dolphin`, `Humpback Whale`, and `Blue Whale` are the species which represents the most individual identifications.**

Checking if the images listed in `train_df` are identical with those found within the list of images in `train_images`:
```python
train_df_list = list(train_df.image.unique())
train_images_list = list(os.listdir('/kaggle/input/happy-whale-and-dolphin/train_images'))
delta = set(train_df_list) & set(train_images_list) # iterable conversion
minus = set(train_df_list) - set(train_images_list) # difference between sets
print(f"Images in train dataset: {len(train_df_list)}\nImages in train folder: {len(train_images_list)}\nIntersection: {len(delta)}\nDifference: {len(minus)}")
```
```output
Images in train dataset: 51033
Images in train folder: 51033
Intersection: 51033
Difference: 0
```
As we can see, there is 100% intersection with 0 differences; indicating that the images listed in `train_df` are identical to the name of the images in the `train_images` folder. 
#
Let's see the range of image dimensions present in the `train_images` dataset.
First we can create a function which returns the shape of a given image filename:
```python
def show_image_size(file_name):
    image = cv2.imread('/kaggle/input/happy-whale-and-dolphin/train_images/' + file_name)
    return list(image.shape)
```

Using a image sample size of **2500** (5% of the dataset), let's set up a new dataframe containing the dimensions of the sample images which will include their `width`, `height`, and `color channels`:
```python
import time
sample_size = 2500
time_alpha = time.time() # start time
train_sample_df = train_df.sample(sample_size)
sample_img_func = np.stack(train_sample_df['image'].apply(show_image_size))
dimensions_df = pd.DataFrame(sample_img_func, columns=['width', 'height', 'c_channels'])
print(f"Total run time for {sample_size} images: {round(time.time()-time_alpha, 2)} sec.")
```
```output
Total run time for 2500 images: 188.83 sec.
```
Now lets include `dimensions_df` to our existing sample dataframe and determine the amount of unique dimensions.
```python
train_img_df = pd.concat([train_sample_df, dimensions_df], axis=1, sort=False)
print(f"Number of different image sizes in {2500} samples: {train_img_df.groupby(['width', 'height','c_channels']).count().shape[0]}")
```
```output
Number of different image sizes in 2500 samples: 1341
```
**NOTE: While there are many different image dimensions concerning height and width, the color channels remain the same throughout the 2500 sample size.**
## Data Preperation
### Data Preperation for EfficientNetB0

Import packages for preprocessing:
```python
import PIL
import PIL.image
import tensorflow as tf
```
Create a list containing all the unique identifications in `train_df`:
```python
id_unique = train_df['individual_id'].unique()
id_unique
```
```output
array(['cadddb1636b9', '1a71fbb72250', '60008f293a2b', ...,
       '3509cb6a8504', 'e880e47c06a4', 'bc6fcab946c4'], dtype=object)
```
Index the list:
```python
id_to_index = dict((name, index) for index, name in enumerate(id_unique))
```
Apply the new index to the images within `train_df`:
```python
image_id_index = [id_to_index[i] for i in train_df['individual_id']]
train_df['label'] = image_id_index
train_df.head()
```
| |`image`|`species`|`individual_id`|`index`|
|-|-------|---------|---------------|-------|
|0| xxxx1.jpg| species_name1| xxxxid1|1|
|1| xxxx2.jpg| species_name2| xxxxid2|2|
|2| xxxx3.jpg| species_name3| xxxxid3|3|
|3| xxxx4.jpg| species_name4| xxxxid4|4|
|4| xxxx5.jpg| species_name5| xxxxid5|5|

Now let's collect the training image file paths for further use:
```python
train_image_paths = ['/kaggle/input/happy-whale-and-dolphin/train_images/' + img for img in train_df['image']]
train_image_paths[:10]
```
```output
['/kaggle/input/happy-whale-and-dolphin/train_images/00021adfb725ed.jpg',
 '/kaggle/input/happy-whale-and-dolphin/train_images/000562241d384d.jpg',
 '/kaggle/input/happy-whale-and-dolphin/train_images/0007c33415ce37.jpg',
 '/kaggle/input/happy-whale-and-dolphin/train_images/0007d9bca26a99.jpg',
 '/kaggle/input/happy-whale-and-dolphin/train_images/00087baf5cef7a.jpg',
 '/kaggle/input/happy-whale-and-dolphin/train_images/000a8f2d5c316a.jpg',
 '/kaggle/input/happy-whale-and-dolphin/train_images/000be9acf46619.jpg',
 '/kaggle/input/happy-whale-and-dolphin/train_images/000bef247c7a42.jpg',
 '/kaggle/input/happy-whale-and-dolphin/train_images/000c3d63069748.jpg',
 '/kaggle/input/happy-whale-and-dolphin/train_images/000c476c11bad5.jpg']
```
Build helper functions to resize the images:
```python
def image_preprocess(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0
    return image
```
```python
def load_and_process(path):
    image = tf.io.read_file(path)
    return image_preprocess(image)
```
Test the function:
```python
for i in range(22):
    temp_img_path = train_image_paths[i]
    temp_label = image_id_index[i]
    plt.imshow(load_and_process(temp_img_path))
    plt.grid(False)
    plt.title(id_unique[i] + " (" + train_df['species'][i] + ")")
```
This will return a sample resized image within the `train_images` folder. You can change the image by changing the number in place of **22** in the function header.
#
### Formatting the data to be used in Tensorflow

First let's generate datasets: `paths_ds`, `images_ds`, `labels_ds`, `image_labels_ds`:
```python
paths_ds = tf.data.Dataset.from_tensor_slices(train_image_paths)
images_ds = paths_ds.map(load_and_process, num_parallel_calls=tf.data.experimental.AUTOTUNE)
labels_ds = tf.data.Dataset.from_tensor_slices(tf.cast(image_id_index, tf.int64))
image_labels_ds = tf.data.Dataset.zip((images_ds, labels_ds))
```
The `image_labels_ds` is a result of binding together (zipping) the `images_ds` and `labels_ds`.

### Dataset tuning
```python
batch_size = 32
ds = image_labels_ds.shuffle(buffer_size=1024)
ds = ds.batch(batch_size)
ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
ds
```
```output
<PrefetchDataset shapes: ((None, 224, 224, 3), (None,)), types: (tf.float32, tf.int64)>
```
#
## EfficientNetB0

**About EfficientNetB0:**
EfficientNet is among the most efficient models, hence its name, that reachers high accuracy on both imagenet and common image classification transfer learning tasks. The base model, EfficientNetB0 provides an efficiency-oriented method of hyperparameter gridsearch while maintaining the least resolution and therefore the least time for modeling. Below, an architecture for the model can be viewed.
![EfficientNetB0 Architecture](https://github.com/UmbertoFasci/Whale_Dolphin_Identification_Competition/blob/main/EfficientNetB0_Arch.png)
This image was found at this [Medium article](https://towardsdatascience.com/complete-architectural-details-of-all-efficientnet-models-5fd5b736142) by Vardan Agarwal.
For more information about EfficientNetB0 and the other EfficientNet models please visit this [website](https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/) which documents them thoroughly.
#
Importing EfficientNetB0
```python
from tensorflow.keras.applications.efficientnet import EfficientNetB0
```
Now we can set up our preprocess_input and our base model:
```python
preprocess_input = tf.keras.applications.efficientnet.preprocess_input

base_model = EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights='imagenet', classifier_activation='softmax')
base_model.trainable=True
prediction_layer = tf.keras.layers.Dense(len(id_unique))
```
```output
Downloading data from https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5
16711680/16705208 [==============================] - 0s 0us/step
16719872/16705208 [==============================] - 0s 0us/step
```
###  Model Setup
```python
inputs = tf.keras.Input(shape=(224, 224, 3))
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
outputs = prediction_layer(x)

model = tf.keras.Model(inputs, outputs)
```
```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```
```python
model.summary()
```
```output
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         [(None, 224, 224, 3)]     0         
_________________________________________________________________
efficientnetb0 (Functional)  (None, 7, 7, 1280)        4049571   
_________________________________________________________________
global_average_pooling2d (Gl (None, 1280)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 1024)              1311744   
_________________________________________________________________
dense_2 (Dense)              (None, 1024)              1049600   
_________________________________________________________________
dense_3 (Dense)              (None, 1024)              1049600   
_________________________________________________________________
dense_4 (Dense)              (None, 1024)              1049600   
_________________________________________________________________
dense (Dense)                (None, 15587)             15976675  
=================================================================
Total params: 24,486,790
Trainable params: 24,444,767
Non-trainable params: 42,023
______________________________________________________________
```
Fit the model:
```python
model.fit(ds, epochs=5)
```
```output
Epoch 1/5

2022-02-05 17:39:06.860953: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
2022-02-05 17:39:18.670639: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:175] Filling up shuffle buffer (this may take a while): 410 of 1024
2022-02-05 17:39:28.667404: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:175] Filling up shuffle buffer (this may take a while): 855 of 1024
2022-02-05 17:39:32.455512: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:228] Shuffle buffer filled.
2022-02-05 17:39:33.976459: I tensorflow/stream_executor/cuda/cuda_dnn.cc:369] Loaded cuDNN version 8005

1595/1595 [==============================] - 1253s 762ms/step - loss: 21.1889 - accuracy: 0.0012
Epoch 2/5

2022-02-05 18:00:40.631584: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:175] Filling up shuffle buffer (this may take a while): 356 of 1024
2022-02-05 18:00:50.583611: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:175] Filling up shuffle buffer (this may take a while): 784 of 1024
2022-02-05 18:00:56.966988: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:228] Shuffle buffer filled.

1595/1595 [==============================] - 1276s 783ms/step - loss: 21.4932 - accuracy: 0.0012
Epoch 3/5

2022-02-05 18:21:56.725817: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:175] Filling up shuffle buffer (this may take a while): 389 of 1024
2022-02-05 18:22:06.698174: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:175] Filling up shuffle buffer (this may take a while): 840 of 1024
2022-02-05 18:22:11.229654: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:228] Shuffle buffer filled.

1595/1595 [==============================] - 1270s 781ms/step - loss: 21.4932 - accuracy: 0.0012
Epoch 4/5

2022-02-05 18:43:18.601180: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:175] Filling up shuffle buffer (this may take a while): 364 of 1024
2022-02-05 18:43:28.595300: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:175] Filling up shuffle buffer (this may take a while): 832 of 1024
2022-02-05 18:43:33.008295: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:228] Shuffle buffer filled.

1595/1595 [==============================] - 1251s 769ms/step - loss: 21.4932 - accuracy: 0.0012
Epoch 5/5

2022-02-05 19:04:40.568634: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:175] Filling up shuffle buffer (this may take a while): 393 of 1024
2022-02-05 19:04:50.549889: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:175] Filling up shuffle buffer (this may take a while): 846 of 1024
2022-02-05 19:04:54.972195: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:228] Shuffle buffer filled.

1595/1595 [==============================] - 1261s 775ms/step - loss: 21.4932 - accuracy: 0.0012

<keras.callbacks.History at 0x7f217576c250>
```
#
## Formatting and Submitting Predictions
First, let's take a look again at the sample_submission.csv:
```python
samp_submission_df.head()
```
```output
 	      image 	                              predictions
0 	000110707af0ba.jpg 	37c7aba965a5 114207cab555 a6e325d8e924 19fbb96...
1 	0006287ec424cb.jpg 	37c7aba965a5 114207cab555 a6e325d8e924 19fbb96...
2 	000809ecb2ccad.jpg 	37c7aba965a5 114207cab555 a6e325d8e924 19fbb96...
3 	00098d1376dab2.jpg 	37c7aba965a5 114207cab555 a6e325d8e924 19fbb96...
4 	000b8d89c738bd.jpg 	37c7aba965a5 114207cab555 a6e325d8e924 19fbb96...
```
Okay! Now that I have taken a good look at how the submissions should look like I will now format the prediction the EfficientNetB0 makes. Firstly, I will make a test dataset:
```python
test_image_paths = ['/kaggle/input/happy-whale-and-dolphin/test_images/' + img for img in samp_submission_df['image']]
test_path_ds = tf.data.Dataset.from_tensor_slices(test_image_paths)
test_image_ds = test_path_ds.map(load_and_process, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_ds = test_image_ds.batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
```
Now I can directly predict on the test dataset I have created `test_ds`:
```python
%%time

pred = model.predict(test_ds)
```
**Predictions are not sufficient. Next I will explore how to look at this as a clustering/similarity problem**
## Next steps
1. Generate image embeddings through model training
2. These embeddings should then be compared with previously generated ones to figure out which clusters the `new_individual` is closest to.
3. Use this general idea to generate predictions.


## Experiments
|**Experiment**|**Description**|**File Link**|
|--------------|---------------|--------|
|Tensorflow ImageDataGenerator|Generate batches of tensor image data with real-time data augmentation. [Reference](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator#flow_from_directory)|[ImageDataGeneratorExperiment](https://github.com/UmbertoFasci/Whale_Dolphin_Identification_Competition/blob/main/experimentalimgdatagen.ipynb)|
|Siamese Network|Generate network containing two or more identical subnetworks to use for image embedding. [Reference](https://keras.io/examples/vision/siamese_network/)|File in Progress|
