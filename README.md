## Project Title
DeepFruitNet: Advanced Deep Learning for Fruit Classification

## Description
This project focuses on developing a deep learning model for classifying various types of fruits. Utilizing the VGG16 and MobileNet architectures, the project addresses the challenges of class imbalance through methods like pruning and class weights adjustment. The model was trained and validated on the ‘Fruits 360’ dataset, achieving high accuracy in fruit classification. The project demonstrates the effective application of convolutional neural networks (CNNs) and transfer learning in image classification tasks.

## Getting Started

### Dependencies
- Python 3.x
- Libraries: 
  - numpy
  - matplotlib
  - seaborn
  - tensorflow
  - sklearn
  - ImageDataGenerator from Keras

### Installation
1. Ensure you have Python 3.x installed on your system.
2. Install the required libraries using pip:
    ```bash
    pip install numpy matplotlib seaborn tensorflow scikit-learn
    ```

### Executing Program
#### Jupyter Notebook Version
1. Open `Advanced Deep Learning for Fruit-opt_project.ipynb` in Jupyter Notebook and run the cells to execute the data preprocessing, model training, and evaluation steps.

#### Python Script Version (if available)
1. Run the script using Python:
    ```bash
    python advanced_deep_learning_for_fruit.py
    ```

### Dataset
The project utilizes the ‘Fruits 360’ dataset, a comprehensive collection of fruit images available for public use. This dataset is sourced from Kaggle and includes a wide variety of fruit images, making it ideal for training and validating the image classification model.

## Contents
- `Advanced Deep Learning for Fruit-opt_project.ipynb`: Jupyter Notebook for the project.
- `Advanced Deep Learning for Fruit-opt_project.pdf`: PDF version of the project report.

### Methodology
#### Data Collection
The ‘Fruits 360’ dataset was employed for this project, containing high-quality images of various fruits categorized into multiple classes.

#### Data Preprocessing
Images were preprocessed to fit the input requirements of the neural network models. This included resizing, normalization, and augmentation techniques such as rotation, width shift, height shift, and horizontal flip.

#### Model Description and Implementation
We explored two CNN architectures: VGG16 and MobileNet. These models were adapted for our fruit classification task using transfer learning techniques. The code for model construction, along with detailed preprocessing steps, is provided in the Appendix of the project report.

#### Training Process
The models were trained on the preprocessed dataset, with a focus on handling class imbalance and optimizing performance. Techniques such as pruning and class weight adjustments were employed during training.

### Results
- **Test Accuracy**: Achieved an accuracy of approximately 98.96% on the test dataset.
- **Test Loss**: Recorded a loss value of 0.0306 on the test dataset.
- A confusion matrix was generated to visually assess the model’s performance across different classes.

### Discussion
The high accuracy of the models indicates their effectiveness in fruit classification tasks. The use of transfer learning with VGG16 and MobileNet architectures significantly contributed to the performance. The challenge of class imbalance was effectively managed through computed class weights. Pruning techniques enhanced the model’s efficiency, making it suitable for deployment in resource-constrained environments.

### Conclusion
This project successfully demonstrates the application of deep learning in fruit classification. The developed models, leveraging advanced CNN architectures, provide a robust solution for accurate and efficient image classification. These models have potential applications in various sectors, including agricultural technology and food quality inspection. Future work may explore further optimization techniques and deployment strategies for real-world applications.

## Authors
- Saroj Raj Amadala
- Srikar Reddy Madireddy

## Version History
- 0.1
    - Initial Release

## References
1. “Recent Advancements in Fruit Detection and Classification Using Deep Learning Techniques,” Hindawi.
2. “Fruit Recognition from Images using Deep Learning Applications,” Springer.
3. “A Hybrid Deep Learning-based Fruit Classification Using Attention Model,” Springer.

## Appendix
The appendix contains detailed code snippets covering various stages of the project, including data preprocessing, model building, training, and evaluation.
