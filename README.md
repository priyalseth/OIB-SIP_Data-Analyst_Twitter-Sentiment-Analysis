#  Twitter Sentiment Analysis

##  Objective
The aim of this project is to develop a machine learning model that can automatically classify tweets into **Positive**, **Negative**, or **Neutral** sentiments. This analysis helps in gaining actionable insights from public opinion, brand perception, or trending topics on social media.

---

##  Overview
This project implements a **Natural Language Processing (NLP)** pipeline for sentiment analysis using the **Twitter_Data.csv** dataset. It follows the complete machine learning workflow from data cleaning to evaluation and prediction.

---

##  Tools & Technologies Used

| Tool/Library        | Purpose                                |
|---------------------|----------------------------------------|
| Python              | Core programming language              |
| Pandas              | Data manipulation                      |
| NLTK                | Text preprocessing (stopwords, stemming) |
| Matplotlib & Seaborn| Data visualization                     |
| WordCloud           | Word cloud generation for insights     |
| Scikit-learn        | Feature extraction, ML modeling, metrics |
| Google Colab        | Cloud-based coding environment         |

---

##  Steps Performed

### 1. **Data Loading & Exploration**
- Imported the dataset and checked structure, missing values, and label distribution.

### 2. **Text Preprocessing**
- Lowercased all text
- Removed URLs, punctuation, special characters, and stopwords
- Applied stemming using NLTK’s `PorterStemmer`

### 3. **Text Visualization**
- Created word clouds for each sentiment category to understand common words

### 4. **Feature Engineering**
- Transformed text data into numerical vectors using `CountVectorizer` (Bag-of-Words model)

### 5. **Model Building**
- Split data into training and testing sets (80/20)
- Trained a **Multinomial Naive Bayes** classifier

### 6. **Model Evaluation**
- Evaluated model performance using:
  - **Accuracy**
  - **Confusion Matrix**
  - **Classification Report**


---

## Outcome

- The model accurately classifies tweet sentiments into Positive, Negative, and Neutral.
- Achieved strong performance using a simple and interpretable Naive Bayes classifier.
- Can be extended for real-time sentiment monitoring, chatbot integration, or feedback analysis.

---

## Sample Visuals
- WordClouds show most frequent words in each sentiment
- Confusion matrix illustrates prediction accuracy per category
---

##  Future Enhancements
- Integrate with **TF-IDF**, **Word2Vec**, or **transformers** like BERT
- Build a **Streamlit/Flask web app** for interactive sentiment checking
- Deploy the model using **FastAPI** or **Docker** for production use

---

##  Author

**Priyal Seth**  
_Data Analyst Intern | Skilled in Python, ML, and NLP_  
sethshreya1999@gmail.com | 🌐 [LinkedIn](https://www.linkedin.com/in/priyal-seth-2493302a2/)

---

