# DeepLabv3plus
## Folder Contents
### _logs_
Stores trained models
### _pretrained_
Pre-trained DeepLabv3+ models
### *weizmann_horse_db*
Dataset and prediction results
## Code Overview and Usage
### Code Functionality
* `annotation.py` is used for segmenting the dataset
* `Mobilnetv2.py` is the backbone
* `model.py` constructs the network model
* `Mydataset.py` prepares the dataset and performs data preprocessing
* `train.py` for training the network
* `predict.py` generates results
* `iou.py` evaluates the results

### Usage Instructions
1.  First, use the `annotation` script to segment the dataset and generate a file containing the names of dataset images.
2.  Then, using the pre-trained model, run the `train` file to generate a trained model, paying attention to the model's output path.
3.  Locate the trained model, run the `predict` script to output results. You can output a single image to see the effect or output the entire dataset.
4.  Finally, evaluate the results using mean Intersection over Union (mIoU) and boundary IoU.

