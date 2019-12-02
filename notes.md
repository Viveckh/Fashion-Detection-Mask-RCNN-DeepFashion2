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