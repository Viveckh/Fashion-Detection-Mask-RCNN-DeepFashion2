### MASK RCNN SETUP

* Clone the Mask_RCNN repo
* Create virtualenv: `virtualenv -p python3 venv`
* Activate the environment
* Install the following specific versions for tensorflow and keras, cause the ones in requirements.txt raises error due to deprecated functions in newer versions
* `pip install tensorflow==1.13.1 keras==2.1.0`
* Install other requirements `pip3 install -r requirements.txt`
* Run MASK RCNN Setup: `python3 setup.py install`
* Download the mask_rcnn_coco.h5 pretrained weights from [here](https://github.com/matterport/Mask_RCNN/releases) and drop in the root of project

### COCO SETUP
* Leave the virtualenv active and clone [this](https://github.com/waleedka/coco) repo separately
* Go to the PythonAPI folder and run the following
* `make`
* `python setup.py install`
* This will install pycoco to the virtualenv


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