# -MACHINE-LEARNING-MODEL-IMPLEMENTATION

COMPANY: CODTECH IT SOLUTIONS

NAME: P SOORYA SHENOY

INTERN ID: CT12ONS

DOMAIN: PYTHON PROGRAMMING

DURATION: 8 WEEKS

MENTOR: NEELA SANTHOSH

DESCRIPTION: This Python program is designed to classify text messages as either spam or ham using Natural Language Processing (NLP) techniques and Machine Learning. It utilizes multiple libraries, including Pandas for data manipulation, NumPy for numerical operations, Seaborn and Matplotlib for visualization, and Scikit-learn for machine learning tasks. The dataset is loaded from a CSV file containing text messages labeled as either "spam" or "ham." To prepare the data, missing values are removed, and the categorical labels are converted into binary values, where "ham" is mapped to 0 and "spam" to 1. The dataset is then split into training and testing sets using the `train_test_split` function, with 80% of the data used for training and 20% for testing. Since machine learning models require numerical inputs, text data is transformed using the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer, which converts text into numerical form while ignoring common stopwords.

After preprocessing, a Multinomial Naïve Bayes classifier, a popular algorithm for text classification, is trained using the transformed training data. The trained model then makes predictions on the test data, and its performance is evaluated using various metrics, including accuracy score, classification report, and confusion matrix. The classification report provides detailed metrics such as precision, recall, and F1-score for each class, helping assess how well the model distinguishes between spam and ham messages. The confusion matrix visually represents the number of correctly and incorrectly classified messages. 

In the provided output, the accuracy of the model is 1.00, indicating perfect classification on the small test set. The classification report shows precision, recall, and F1-score values of 1.00 for both classes, meaning all predictions were correct. However, a warning appears in the output regarding the shape of the confusion matrix, suggesting that explicit label definitions should be provided to avoid issues when evaluating the model. This result suggests that the dataset might be too small for generalization, and testing on a larger dataset is recommended to ensure robust performance.

OUTPUT:
![image](https://github.com/user-attachments/assets/dbc9bc61-85cd-4960-bba2-63497f8948d7)
![image](https://github.com/user-attachments/assets/bc6097c1-e26f-4856-bc05-8d15813da7ff)

