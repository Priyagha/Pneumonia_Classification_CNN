# Pneumonia Classification from X-Ray Image using CNN

Pneumonia is a common infectious disease that is responsible for over 1 million cases and tens of thousands of deaths annually in the US alone.As an example, in 2017, 1.3 million cases were reported, of which over 50,000 died, resulting in a death rate of nearly 4%.Timely detection of pneumonia is critical as it can be life-threatening if left untreated. This project aims to detect the presence of Pneumonia in individauls based on the X-Ray images using Convolutional Neural Network(CNN).

## Dataset

The data is sourced from **Arizona Pneumonia Detection Challenge dataset** provided by the Radiological Society of North America. It consists of **26,684** X-ray images, of which about 20,000 do not show signs of pneumonia. And around 6000 actually do show signs of pneumonia. The data's inherent imbalance poses a significant challenge that we address during the model development process.

The labels indicate whether they depict cases of pneumonia or not and are stored in `stage_2_train_labels.csv`. The dataset's distribution reflects the real-world prevalence of pneumonia cases, with a substantial majority of images portraying healthy individuals. This imbalanced distribution necessitates careful handling to ensure model performance and generalization.

## Method

We utilized the **ResNet18** architecture to classify X-ray images into pneumonia-positive or not. CNNs are particularly well-suited for image classification tasks due to their ability to extract hierarchical features from input images. Our model architecture comprises multiple convolutional and pooling layers followed by fully connected layers and output nodes for binary classification. We utilize techniques such as data augmentation, transfer learning, and model fine-tuning to enhance model performance and mitigate the effects of data imbalance.

**Data Preprocessing**: The first step involved preprocessing the dataset, including resizing images from **1024 X 1024 to 224 X 224**, This was necessary as 1024 is way to large for current machine learning algorithms. Next, we standardize all pixel values into the zero one interval by scaling them with a constant factor of one divided by 255, and finnaly splitting into training, and validation sets. We ensured uniformity in image dimensions to facilitate model training. Additionally we compute mean and standard deviation of the training dataset for the purpose of normalization.

**Model Architecture Selection**: We utilize the ResNet-18 architecture for pneumonia classification, adapting it to handle grayscale images by changing the number of input channels from 3 to 1. The model is trained using the PyTorch Lightning library, which provides high-level abstractions for effective training. We employ techniques such as data augmentation, including random rotations, translations, scales, and resized crops, to enhance model robustness.

**Transfer Learning**: To leverage the pre-trained weights of ResNet-18 on large-scale image datasets such as ImageNet, we employed transfer learning. By initializing our model with pre-trained weights, we accelerated the convergence process and improved the model's ability to extract relevant features from X-ray images.

**Fine-tuning**: We fine-tuned the pre-trained ResNet-18 model on the pneumonia detection task using our dataset. Fine-tuning involved updating the model's weights through backpropagation while adjusting the learning rate and employing techniques like gradient clipping to prevent overfitting.

**Data Augmentation**: To address the data imbalance and enhance model generalization, we applied data augmentation techniques such as rotation, horizontal/vertical flipping, and zooming. Data augmentation artificially increases the dataset size and diversity, enabling the model to learn robust features from limited data.

**Model Training**: We trained the ResNet-18 model on the preprocessed dataset using **Adam Optimizer**. During training, we monitored key performance metrics such as loss and accuracy on both training and validation sets to gauge model convergence and identify potential issues like overfitting.

**Model Evaluation**: After training, we evaluated the trained model's performance on the validation set, assessing metrics such as precision, recall, and F1-score. Due to class imbalance we cannot rely on accruarcy as a model evaluation matrix. Additionally, we conducted qualitative analysis using techniques like confusion matrix visualization to gain insights into the model's behavior and identify potential areas for improvement. 

## Conclusion

In summary, our project aims to develop an accurate and interpretable pneumonia classification model from X-ray images. By leveraging state-of-the-art techniques in deep learning and image analysis, we strive to contribute to the early detection and management of pneumonia, ultimately improving patient outcomes and reducing mortality rates.