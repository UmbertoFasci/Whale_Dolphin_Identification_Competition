# Whale Dolphin Identification Competition
> Kaggle competition (Happywhales - Whale and Dolphin Identification)

**Contents**

 * [Overview](#overview)
 * [Data Exploration](#data-exploration)
 * [Data Preperation - CNN](#data-preperation---cnn)
 * [Convolutional Neural Network Model (CNN)](#convolutional-neural-network-model-cnn)
 * [More Models](#more-models)

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
![Distribution of Species Individual IDs - Training Set](https://github.com/UmbertoFasci/Whale_Dolphin_Identification_Competition/blob/main/Distribution_of_Species_imgs.png)

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
![Distribution of Species Images - Training Set](https://github.com/UmbertoFasci/Whale_Dolphin_Identification_Competition/blob/main/Distribution_of_Species_ID.png)

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

Using a image sample size of **25000** (5% of the dataset), let's set up a new dataframe containing the dimensions of the sample images which will include their `width`, `height`, and `color channels`:
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
## Data Preperation - CNN
**In Progress**

Considering:
* Tensorflow ImageDataGenerator

## Convolutional Neural Network Model (CNN)
**In Progress**

## More models
**In Progress**
