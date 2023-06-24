import numpy as np
import os
from keras.models import load_model
from constants import test_image_dir
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize

# Load the trained segmentation model
seg_model = load_model('seg_model.h5', compile=False)

# Load test data from the folder
test_image_files = sorted(os.listdir(test_image_dir))[:100]

#make predictions and save them to np.array
predictions = []
for c_img_name in test_image_files:
    c_path = os.path.join(test_image_dir, c_img_name)
    c_img = imread(c_path)
    c_img_resized = resize(c_img, (768, 768), preserve_range=True)
    c_img = np.expand_dims(c_img_resized, 0)/255.0
    cur_seg = seg_model.predict(c_img)[0]
    predictions.append(cur_seg)

predictions = np.array(predictions)

# Perform further analysis or visualization with the predictions
def analyze_predictions(predictions, threshold=0.5):
    """
    Analyzes the predictions and computes the statistics.

    Args:
        predictions (ndarray): Array of predictions.
        threshold (float): Threshold value for classifying predictions as positive or negative.

    Returns:
        dict: Dictionary containing the analysis results.
    """
    # Apply threshold to convert predictions to binary values
    binary_predictions = (predictions > threshold).astype(int)

    # Compute the number of positive and negative predictions
    num_positive = np.sum(binary_predictions)
    num_negative = len(binary_predictions) - num_positive

    # Compute the average prediction value
    average_prediction = np.mean(predictions)

    # Create a dictionary to store the analysis results
    analysis_results = {
        "num_positive": num_positive,
        "num_negative": num_negative,
        "average_prediction": average_prediction
    }

    return analysis_results


# Call analyze_predictions
analysis_results = analyze_predictions(predictions, threshold=0.5)

# Print the analysis results
print("Number of positive predictions:", analysis_results["num_positive"])
print("Number of negative predictions:", analysis_results["num_negative"])
print("Average prediction value:", analysis_results["average_prediction"])

fig, m_axs = plt.subplots(8, 2, figsize = (10, 40))
for (ax1, ax2), c_img_name in zip(m_axs, test_image_files):
    c_path = os.path.join(test_image_dir, c_img_name)
    c_img = imread(c_path)
    c_img_resized = resize(c_img, (384, 384), preserve_range=True)
    first_img = np.expand_dims(c_img_resized, 0)/255.0
    first_seg = seg_model.predict(first_img)
    ax1.imshow(first_img[0])
    ax1.set_title('Image')
    ax2.imshow(first_seg[0, :, :, 0], vmin = 0, vmax = 1)
    ax2.set_title('Prediction')
fig.savefig('test_predictions.png')
plt.show()


