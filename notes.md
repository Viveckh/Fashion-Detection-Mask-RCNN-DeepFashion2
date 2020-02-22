### BLU Image Instance Segmentation Model Training

## Author: [(EJ) Vivek Pandey](https://viveckh.com)

### MASK RCNN SETUP

* If you're launching an EC2 instance, start with `g4dn.xlarge` - about $0.5 an hour at the time of writing this.
* Also use an Ubuntu deep learning instance, since I had some cython issues while building pycocotools in the Amazon Linux instance.
* Clone the Mask_RCNN repo
* Install virtualenv if you dont have it: `pip3 install --user virtualenv`
* Create virtualenv: `virtualenv -p python3 venv`
* Activate the environment
* Install the following specific versions for tensorflow and keras, cause the ones in requirements.txt raises error due to deprecated functions in newer versions
* `pip3 install tensorflow==1.13.1 keras==2.1.0`
* Install other requirements `pip3 install -r requirements.txt`
* Run MASK RCNN Setup: `python3 setup.py install`
* Download the mask_rcnn_coco.h5 pretrained weights from [here](https://github.com/matterport/Mask_RCNN/releases) and drop in the root of project. Or you can `scp` it from your local if you're on cloud

### COCO SETUP
* Leave the virtualenv active and clone [this](https://github.com/waleedka/coco) repo in a separate folder
* Go to the PythonAPI folder and run the following
* `easy_install cython` only if you get cython error
* `make`
* `python setup.py install`
* This will install pycocotools to the virtualenv. Verify by running a `pip freeze`

### Optional: Testing the setup: Running Coco evaluation in samples

The following assumes you have already downloaded the 2017 validation image and annotation sets into the `/CodeForLyf/_Datasets/coco` folder.

`python3 coco.py evaluate --dataset "/CodeForLyf/_Datasets/coco" --year 2017 --model "/CodeForLyf/Mask_RCNN/mask_rcnn_coco.h5" --limit 10`

### Optional: Blu Model Training by starting from coco pretrained model if on AWS
Navigate to the folder with `blu.py`

`python3 blu.py train --dataset "s3://deepfashion2" --model "coco" --usecachedannot false --limit 10`

Remove the limit of 10 once you verify it runs

# Quirks on AWS

### If you get an `AssertionError` that says something along the lines of keras backend not finding a match from {'tensorflow', 'theano' and 'cntk'}

`vi ~/.keras/keras.json`

And update the backend to tensorflow, since the default might be set to MXNET.

### If you get an error pertaining to the channel dimensions of the inputs not being defined

`vi ~/.keras/keras.json`

Update the `image_data_format` to `channels_first`. Since we are attempting to use pretrained weights, it should be loaded from there.

### Detection Response
The detection returns the following dict for each image.
* rois: [N, (y1, x1, y2, x2)] detection bounding boxes

```python
array([[216, 332, 299, 423],
       [141, 364, 222, 455],
       [216, 427, 302, 521],
       [ 85, 419, 160, 502],
       [144, 182, 223, 268],
       [215, 228, 306, 327],
       [148, 270, 226, 357],
       [ 85, 224, 157, 305],
       [ 80, 311, 151, 388],
       [120, 117, 361, 174],
       [ 11,  18, 394, 635]], dtype=int32)
```
* class_ids: [N] int class IDs

```python
array([55, 55, 55, 55, 55, 55, 55, 55, 55, 44, 61], dtype=int32)
```

* scores: [N] float probability scores for the class IDs

```python
array([0.9992748 , 0.9991703 , 0.9991478 , 0.99901104, 0.9988919 ,
       0.9988857 , 0.9984559 , 0.9981918 , 0.99819165, 0.9757698 ,
       0.96571267], dtype=float32)
```
* masks: [H, W, N] instance binary masks

```python
    # So if there are 11 instances found in a 394 x 640 image, this is what the shape of the mask looks like.
    (394, 640, 11)
    >>> masks[216][332]
    array([False, False, False, False, False, False,  True, False, False,
       False,  True])
    # For every image pixel, it is tracked which of the instances it contains through the boolean mask. So ideally, if you traverse through the tensor as masks[x][y][0], you should get the mask for the first instance.
```