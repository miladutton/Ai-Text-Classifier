# 🧠 AI vs Human Text Classifier

A machine learning project that classifies text as either **AI-generated** or **human-written** using TF-IDF features and linear models.  
The dataset comes from [Kaggle - AI vs Human Text Dataset](https://www.kaggle.com/datasets/shamimhasan8/ai-vs-human-text-dataset).

## 📌 Overview
This project:
- Loads and cleans the dataset
- Converts text into numerical features using **TF-IDF**
- Trains and evaluates **Logistic Regression** and **Linear Support Vector Classifier (SVC)**
- Tunes hyperparameters with **GridSearchCV**
- Visualizes results with a confusion matrix
- Shows most informative n-grams for each class
- Saves the trained model for future use


## 📥 Dataset
Download from Kaggle: [AI vs Human Text Dataset](https://www.kaggle.com/datasets/shamimhasan8/ai-vs-human-text-dataset)  
Place the file in the project root as:
\`\`\`
ai_vs_human_text.csv
\`\`\`

## 🚀 Usage
1. Open the Jupyter notebook:
\`\`\`bash
jupyter notebook Ai_vs_Human_Text_Classifier_Portfolio.ipynb
\`\`\`
2. Run all cells to:
   - Train models
   - View evaluation metrics
   - Save the best model

3. Use the saved model for predictions:
\`\`\`python
import joblib
model = joblib.load("ai_human_text_detector.joblib")
model.predict(["This is a sample text to classify."])
\`\`\`

## 📊 Example Results
- **Baseline Accuracy (LinearSVC)**: ~0.95+
- Confusion matrix shows clear separation between AI and human texts
- Top n-grams reveal distinct stylistic patterns

## 🛠 Next Steps
- Deploy with Gradio for interactive predictions
- Experiment with transformer embeddings
- Test robustness against paraphrasing and style transfer

## 📄 License
This project is licensed under the MIT License.
