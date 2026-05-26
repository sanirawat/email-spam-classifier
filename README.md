# Email Spam Classifier

An intelligent machine learning-based application that classifies emails/messages as **Spam** or **Not Spam (Ham)** using Natural Language Processing (NLP) techniques and machine learning algorithms.


## 🚀 Project Overview

This project is designed to detect whether an email or message is spam or legitimate.  
It uses text preprocessing, TF-IDF vectorization, and machine learning models to make accurate predictions.

The classifier analyzes the text content and predicts:
- ✅ Not Spam (Ham)
- 🚫 Spam


## Features

- 📩 Spam and Ham message classification
- 🧠 Machine Learning based prediction
- 📝 NLP text preprocessing
- ⚡ Fast and accurate predictions
- 💾 Saved trained model using Pickle
- 📊 Simple and user-friendly interface


##  Tech Stack

- Python
- Scikit-learn
- Pandas
- NumPy
- NLTK
- TF-IDF Vectorizer
- Streamlit


## Project Structure

```bash
email-spam-classifier/
│
├── app.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── spam.csv
├── notebook.ipynb
└── README.md
```


## How It Works

1. User enters an email/message
2. Text preprocessing is applied:
   - Lowercasing
   - Removing punctuation
   - Tokenization
   - Stopword removal
   - Stemming/Lemmatization
3. Text is converted using TF-IDF Vectorization
4. The trained model predicts whether the message is spam or not



## Installation

### Clone the Repository

```bash
git clone https://github.com/sanirawat/email-spam-classifier.git
cd email-spam-classifier
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Project

### For Streamlit

```bash
streamlit run app.py
```

### For Flask

```bash
python app.py
```

---

##  Machine Learning Workflow

- Data Collection
- Data Cleaning
- Text Preprocessing
- Feature Extraction using TF-IDF
- Model Training
- Model Evaluation
- Prediction



##  Algorithms Used

- Naive Bayes
- Logistic Regression
- Support Vector Machine (SVM)

---

## 🎯 Future Improvements

- Deep Learning based spam detection
- Real-time email integration
- Multi-language support
- Cloud deployment
- Advanced NLP models

---

## Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Push the branch
5. Open a Pull Request

---

##  License

This project is licensed under the MIT License.

---

## 👩‍💻 Author

**Saniya Rawat**

GitHub: https://github.com/sanirawat
