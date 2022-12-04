# KMNIST_Cluster

Do clustering on images without labels, and then evaluate the clustering result by criterion NMI (Normalized Mutual Information) and ARI (Adjusted Rand Index).

## Requirements for package

After download the codes, you can install the required python packages by running

```
$ pip install -r requirements.txt
```

## Usage

### 1. Dataset

As the title mentioned, this is for KMNIST(10 classes), for more details about this dataset, please refer to 

<https://github.com/rois-codh/kmnist>

### 2. Convert the ubyte file to images and import them
```
    # Here is the original dataset location
    train_images = 'THE ORIGINAL DATASET LOCATION'
    train_labels = 'THE ORIGINAL DATASET LOCATION' # only import it, but do not use it
    test_images = 'THE ORIGINAL DATASET LOCATION' # only import it, but do not use it
    test_labels = 'THE ORIGINAL DATASET LOCATION' # only import it, but do not use it
    
    # Here is the converted dataset location
    save_train = 'THE CONVERTED DATASET LOCATION'
    save_test = 'THE CONVERTED DATASET LOCATION'
```

### 3. Principal component analysis

Before clustering, reduce the dimension of the giant matrix. You can see the variance contribution rate of PCA.

### 4. K-means clustering

You can try clustering on the matrix after PCA by

```
# cluster on the matrix after PCA
km.fit(pca.transform(A))
```

Of course, you can try clustering on the original matrix by

```
# cluster on the original matrix
km.fit(X_reshape)
```

### 5. Save the result of the clustering
#### (a) Create a specific file
#### (b) Change the path you want to save
The core code is:

```
image1.save('THE PATH YOU WANT TO SAVE' + 'cluster1_' + str(j + 1) + '.jpg')
```

### 6. Calculate ARI and NMI of the clustering result

Calculate ARI and NMI of the clustering result under different dimensions after PCA and plot line charts. Obtain the best dimension after PCA dimension reduction.

# KMNIST_Classification

Doing KMNIST Classification using Pytorch

## Requirements for package

After download the codes, you can install the required python packages by running

```
$ pip install -r requirements.txt
```

## Usage

### 1. entry the directory

```
$ cd path-to-codes
```

### 2. Dataset

As the title mentioned, this is for KMNIST(10 classes), for more details about this dataset, please refer to 

<https://github.com/rois-codh/kmnist>

### 3. Train

When train a specified model, please use arg -net

```
#training resnet18 on gpu
$ python train.py -net res18 -gpu
```
The detailed usage is:

```
$ train.py [-h] -net NET [-gpu] [-bs BS] [-Epoch EPOCH] [-lr LR] [-wd WD] [-momentum MOMENTUM] [-resize RESIZE] [-num_workers NUM_WORKERS] [-aug] [-path PATH] [-name NAME]

```
The path is where you store your training results, the whole store path will be path/net, name is the file name of the results. Here is one example

```
python train.py -net res18 -gpu -path ./runs -name train
```
Then you will get a directory in the following template in the current workspace

```
runs
└── res18
    ├── last.pt
    ├── train.csv
    └── train.jpg
```


Current available networks are:(to be countinued)

```
-mlp
-lenet
-alexnet
-vgg11
-vgg13
-vgg16
-vgg19
-googlenet
-res18
-res34
-densenet121
-densenet169
-densenet201
-densenet264

```

### 4. test

After you finished training, use args -net to test it on test set, here just use the test set again.

```
#test resnet18 we have just trained before
$ python test.py -net res18 -gpu
```

the detailed usage is:

```
$ test.py [-h] -net NET [-gpu] [-bs BS] [-resize RESIZE] [-num_workers NUM_WORKERS] [-path PATH] [-name NAME]
```
the usage of path and name are the same as those in train.py. After finish testing, you will get a figure named '{your name}.jpg',
this is a confusion matrix for your testing result.

### 5. predict 
#### (a) 
If you have your own test sets, and put them into different directories 
(make sure the images with the same label are put in the same directory whose name is just the label)
Then you can testing on the set of yourself by:

```
$ python predict.py -net res18 -gpu -all
```

the detailed usage is 

```
$ predict.py [-h] -net NET [-gpu] [-image IMAGE] [-all] [-bs BS] [-resize RESIZE] [-num_workers NUM_WORKERS] [-path PATH] [-path_dataset PATH_DATASET] [-name NAME]
```
After that, you will get a comfusion matrix image named '{your name}.jpg' in the path you set

#### (b) 
If you don't have your own dataset to do testing, you can only use it to predict one image at once, then just run as below

```
$ python predict.py -net res18 -gpu -image [path-to-image/image-name]
```

then it will show the prediction results just like this

![image](https://github.com/huangdyu/KMNIST_Classification/blob/main/Sample.png)














