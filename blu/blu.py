"""
Model training and evaluation codebase for a Mask-RCNN Network on DeepFashion Dataset
- Converts the deepfashion dataset and annotations to a coco-compatible format

Written by (EJ) Vivek Pandey. Baseline by Abdulla Waleed
www.github.com/Viveckh

------------------------------------------------------------

TODO: Update Instructions

python3 blu.py train --dataset "/CodeForLyf/_Datasets/deepfashion2" --model "/CodeForLyf/Mask_RCNN/mask_rcnn_coco.h5" --usecachedannot true --limit 10
Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from scratch using deepfashion dataset
    python3 blu.py train --dataset "</path/to/deepfashion>" --model "deepfashion" --usecachedannot false --limit 10

    # Train a new model starting from pre-trained COCO weights
    python3 blu.py train --dataset "</path/to/deepfashion>" --model "coco" --usecachedannot false --limit 10

    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet --download=True

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import sys
import time
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Import custom written utilities
from deepfashion2_to_coco import convert_deepfashion2_annotations_to_coco_format

# Path to trained weights file
# Update the default model path
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
DEEPFASHION_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_deepfashion2.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2014"


# TODO: Typecast the functions
############################################################
#  Configurations
############################################################

class BluConfig(Config):
    """Configuration for training on Blu Data
    Derives from the base Config class and overrides values specific
    to the Blu dataset
    """
    # Give the configuration a recognizable name
    NAME = "blu"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    # Changing the following from COCO's 80 classes makes models trained on COCO unusable unless you ignore last layer of NN while loading weights
    NUM_CLASSES = 1 + 13  # The Deepfashion2 dataset being currently used has 13 classes


############################################################
#  Dataset
############################################################
class BluDataset(utils.Dataset):
    def load_deepfashion2(self, dataset_dir, subset, use_cached_coco_annot=False, return_coco=True):
        """Load a subset of the deepfashion2 data
        dataset_dir: The root directory of the deepfashion2 dataset.
        subset: What to load (train, validation)
        use_cached_coco_annot: Use cached coco-style annotations previously generated
        return_coco: If True, returns the COCO-style object.
        """

        temp_json_annotations_output = "temp/deepfashion_{}_annotations.json".format(subset)
        
        if use_cached_coco_annot and os.path.isfile(temp_json_annotations_output):
            pass
        else:
            # First transform the annotations from deepfashion to COCO format and drop it in a file
            if not convert_deepfashion2_annotations_to_coco_format(dataset_dir, subset, temp_json_annotations_output):
                raise Exception("Deepfashion annotations could not be converted to coco-style")

        # Then initialize COCO with the coco-style annotation file
        deepfashion = COCO(temp_json_annotations_output)
        image_dir = "{}/{}/image".format(dataset_dir, subset)

        # Load all classes (or a subset if you can filter train/test per category at some point)
        class_ids = sorted(deepfashion.getCatIds())

        # Load all images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(deepfashion.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(deepfashion.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("deepfashion", i, deepfashion.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                source="deepfashion", image_id=i,
                path=os.path.join(image_dir, deepfashion.imgs[i]['file_name']),
                width=deepfashion.imgs[i]["width"],
                height=deepfashion.imgs[i]["height"],
                annotations=deepfashion.loadAnns(deepfashion.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return deepfashion

    # TODO: Implement an auto-download feature in case the dataset isn't already in the system

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a Deepfashion image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "deepfashion":
            return super(BluDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            # The map_source_class_id is created in the .prepare() function of parent Utils class which is not overwritten
            class_id = self.map_source_class_id(
                "deepfashion.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(BluDataset, self).load_mask(image_id)

    
    # The following two functions are from pycocotools with a few changes.
    
    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


############################################################
#  Deepfashion Training
############################################################

def train(model, config, dataset_dir: str, use_cached_coco_annot: bool):
    # Training dataset.
    dataset_train = BluDataset()
    dataset_train.load_deepfashion2(dataset_dir=dataset_dir, subset='train', use_cached_coco_annot=use_cached_coco_annot, return_coco=True)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = BluDataset()
    dataset_val.load_deepfashion2(dataset_dir=dataset_dir, subset='validation', use_cached_coco_annot=use_cached_coco_annot, return_coco=True)
    dataset_val.prepare()

    # Image Augmentation
    # Right/Left flip 50% of the time
    augmentation = imgaug.augmenters.Fliplr(0.5)

    # Training - Stage 1
    # Only training the network heads since the initial training will be on top of coco pretrained model
    # TODO: Update this to handle case where you're training from scratch
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                layers='heads',
                augmentation=augmentation)

    """
    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=120,
                layers='4+',
                augmentation=augmentation)

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=160,
                layers='all',
                augmentation=augmentation)
    """

############################################################
#  Deepfashion Evaluation
############################################################

def build_deepfashion_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id=class_id, source="deepfashion"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_deepfashion(model, dataset, deepfashion, eval_type="bbox", limit=0, image_ids=None):
    """Runs deepfashion evaluation.
    dataset: A Dataset object with validation data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick deepfashion images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding deepfashion image IDs.
    deepfashion_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_deepfashion_results(dataset, deepfashion_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    deepfashion_results = deepfashion.loadRes(results)

    # Evaluate using cocoEval
    cocoEval = COCOeval(deepfashion, deepfashion_results, eval_type)
    cocoEval.params.imgIds = deepfashion_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


############################################################
#  Entry point
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on DeepFashion2.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on Deepfashion")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/deepfashion2/",
                        help='Directory of the Deepfashion2 dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco', 'deepfashion' or 'last' to continue from last trained weights")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    parser.add_argument('--usecachedannot', required=False,
                        default=False,
                        metavar="<True|False>",
                        help='Automatically uses the cached COCO-style annotations if available (default=False)',
                        type=bool)
    parser.add_argument('--download', required=False,
                        default=False,
                        metavar="<True|False>",
                        help='Automatically download and unzip Deepfashion files (default=False)',
                        type=bool)
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    print("Use Cached COCO Annotations: ", args.usecachedannot)
    print("Auto Download: ", args.download)

    # Configurations
    if args.command == "train":
        config = BluConfig()
    else:
        class InferenceConfig(BluConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "deepfashion":
        model_path = DEEPFASHION_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    # If you wanna use COCO's pretrained weights, changing the NUM_CLASSES in BluConfig to Deepfashion no of classes causes issues
    print("Loading weights ", model_path)
    if args.model.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes. Without this, the changes in config will raise errors during training 
        model.load_weights(model_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(model_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model=model, config=config, dataset_dir=args.dataset, use_cached_coco_annot=args.usecachedannot)

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = BluDataset()
        deepfashion = dataset_val.load_deepfashion2(dataset_dir=args.dataset, subset='validation', use_cached_coco_annot=args.usecachedannot, return_coco=True)
        dataset_val.prepare()

        print("Running Deepfashion evaluation on {} images.".format(args.limit))
        evaluate_deepfashion(model, dataset_val, deepfashion, "bbox", limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
