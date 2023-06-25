# Segmentation_Unet_Test_task
Solving test task for computer vision using Unet
## Project description
This project focuses on solving image segmentation problems using semantic segmentation technique. The project utilizes the popular U-Net architecture implemented with TensorFlow's Keras API. The Dice score is used as the evaluation metric to assess the model's performance. Model build based on high-scored solution published on kaggle (https://www.kaggle.com/code/kmader/baseline-u-net-model-part-1).

## Project Structure

The project is organized as follows:

- \data\train_v2\: Training data images - need to be uploaded before run
- \data\test_v2\: Test data images - need to be uploaded before run
- \data\train_ship_segmentations_v2.csv: Encoded data with ships masks
- `dataloader.py`: Python script for data preprocessing and augmentation.
- `constants.py`: Constants and variables used throughout the project.
- `model_training.py`: Python script for training the semantic segmentation model.
- `model_inference.py`: Python script for performing model inference on new images.
- `data_prep.ipynb`: Jupyter Notebook for exploring the dataset and analyzing the image data.
- `requirements.txt`: File listing the required Python modules for easy installation.
- 'seg_model.h5': Trained model

## Project details

To solve the image segmentation problem, the following steps were performed:

1. **Data Preprocessing**: The dataset was preprocessed to prepare it for training the model. This included encoding/decoding, droping images without ships to improve imbalance, normalizing pixel values, and handling missing or corrupted data. The `dataloader.py` script was used to perform these tasks.

2. **Data Augmentation**: Data augmentation techniques were applied to increase the diversity of the training data and improve the model's ability to generalize. Augmentation techniques such as random rotations, flips, and shifts were used to create additional training samples with the `dataloader.py` script.

3. **Exploratory Data Analysis**: The `data_prep.ipynb` Jupyter Notebook was used to perform an analysis of the dataset. This involved viewing and visualizing maps, examining the distribution of ships.

4. **Model Training**: The `model_training.py` script was used to train the U-Net model on the preprocessed and augmented dataset. The script allowed for customization of hyperparameters, such as learning rate, batch size, and number of training epochs. The model was trained using a combination of loss functions and the Dice score was used to evaluate its performance. Trained model saved in separate file 'seg_model.h5'.

5. **Model Inference**: The trained model was used for performing inference on new images using the `model_inference.py` script. Given an input image, the script applied the segmentation model to generate pixel-level predictions, producing segmentation masks for the objects of interest.


## Conclusion
This project provides a solution for image segmentation tasks using the U-Net architecture and TensorFlow's Keras API. The code  includes separate files for data preprocessing, model training, and model inference. The Jupyter Notebook facilitates data exploration and analysis, aiding in model development and understanding the dataset. Achieved dice-score is about 0.79.
