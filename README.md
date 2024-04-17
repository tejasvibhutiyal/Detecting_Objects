#Object Detection Project
This project focuses on detecting objects including "person", "glasses", "pen", and "bottle" using a custom-trained YOLOv5 model. 
The dataset utilized for training was collected from Roboflow, specifically curated to include images of these objects, with special attention given to images containing people wearing glasses and without glasses to accurately differentiate between "glasses" and "person".

##Dataset
The dataset contains images of various objects, with a particular emphasis on people wearing glasses and without glasses. This diversity ensures robustness in the model's ability to distinguish between different classes. For access to the preprocessed dataset used in this project, refer to the following link on Hugging Face: Glasses and Pens Merged Dataset.

##Classes
The following are the classes that the model is trained to detect:

"person" (Class index: 0)
"glasses" (Class index: 1)
"bottle" (Class index: 2)
"pen" (Class index: 3)
The model has been trained to identify these objects which are not available in yolov5 model.
