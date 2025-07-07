## Analytics module for correction and feedback of long jump athlete using anomaly detection
### Pradyumn Vikram

## Usage
Install required dependancies by running\
```pip install mediapipe opencv-python scikit-learn numpy```\
Run the module via\
```python -m analytics_module.py```\

## Directory Structure
- The main executable for this projecy is analytics_module.py, which takes a trimmed long jump sequence as input, giving corrective feedback as output\
- generator_trimmer.ipynb contains the pipeline for extracting the dataset and processing it for phase segmentation - it requries the data directory with altered_codec, final_dataset, processed, raw_data, ref_vid, split_videos sub directories - these have not been included in the repo since the content is downloaded by the notebook automatically\Note: Convert the downloaded videos (saved in raw_data) to a suitable codec for processing and transfer them to the altered_codec directory\
- preprocessing_pose_extraction.ipynb contains the pipeline for extracting clusters and segmenting phases\
- train_extracted_features_anomaly_detection.ipynb contains the pipeline for training the final anomaly detection model\
- All models are saved in models/ directory and the scalers are saved in scalers/\
- project report and test videos can be found in the misc folder\
- keypoints contains extracted feature arrays which are processed furthur for anomaly detection\

## Project Report
A report for the project can be found [here](https://github.com/PradyumnVikram/ProjectKitty/blob/main/misc/report.pdf)
