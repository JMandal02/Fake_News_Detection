# Fake_News_Detection
This project is a complete Fake News Detection system combining Machine Learning and a responsive web interface using Streamlit.Users can input news articles and instantly check if they are Fake or Real — with a confidence score and a simple explanation.



## Demo 🚀
Live Deployment:[https://fakenews-analyzer.streamlit.app/](https://fakenews-analyzer.streamlit.app/)



## Tech Stack 🛠️
- Programming Language: Python 3
- Frontend Framework: Streamlit
- Machine Learning Libraries: Scikit-learn, NLTK, Pandas, NumPy
- Model Persistence: Pickle (.pkl files)
- Styling: Custom CSS for Streamlit



## Project Workflow (Technical Flowchart) 🔥
```bash
START
  ↓
Install Python Libraries
  - streamlit, scikit-learn, pandas, numpy, nltk
  ↓
Download NLTK Resources
  - punkt, stopwords, wordnet
  ↓
Load Dataset (from Kaggle)
  ↓
Data Preprocessing
  - Lowercasing
  - Punctuation Removal
  - Tokenization
  - Stopwords Removal
  - Lemmatization
  ↓
Text Vectorization
  - Using CountVectorizer or TfidfVectorizer
  ↓
Split Dataset
  - Training set and Testing set
  ↓
Train Machine Learning Model
  - Logistic Regression / Random Forest / Any classifier
  ↓
Evaluate Model
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  ↓
Save Trained Model
  - Export as `.pkl` file using pickle
  ↓
Build Streamlit Web Application
  - User input
  - Load Model
  - Preprocess input text
  - Predict
  - Display Prediction 
  ↓
Deploy Web App
  - GitHub + Streamlit Cloud
END
```

## Installation Instructions ⚙️
### 1. Install Python and pip 🐍
- Download and install Python (version 3.8 or higher) from python.org.
- During installation, make sure to check "Add Python to PATH" option.
#### Verify installation:
```bash
python --version
pip --version
```

### 2. Install Required Libraries Manually 📚
Now you can install all libraries with just one command using requirements.txt:
```bash
pip install -r requirements.txt
```
Also download NLTK resources manually:
```bash
pip install streamlit
pip install scikit-learn
pip install pandas
pip install numpy
pip install nltk
```

### 3. Download Dataset from Kaggle 📥
- Create a free account on Kaggle.
- Find two datasets for fake news detection (e.g. "Fake News" , " True news").
- Download the dataset CSV files.
- Place it in your project's dataset/ folder.



## Download required NLTK resources inside your Python environment:
```bash
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```



## How to Run the Application Locally 🖥️
- After installing all dependencies, inside your cmd:
```bash
streamlit run app.py
```
### It will automatically open in your web browser at:
```bash
http://localhost:digits
```
- Input news text → Click Predict → See Result, Confidence Score, and Explanation!



## Deployment Instructions ☁️
You can easily deploy this application using Streamlit Community Cloud:
- Push all project files (including requirements.txt) to GitHub.
- Go to Streamlit Cloud and sign in.
- Click "New App", select your GitHub repository.
- Select the main Python file (app.py).
- Click Deploy!
You will get a public HTTPS link to share your app with everyone.


## Project Structure 🗂️
```bash
├── app.py               # Main Streamlit app
├── model.pkl            # Saved ML model
├── requirements.txt     # Python dependencies
├── vectorizer.pkl       # Saved ML model
├── dataset/             # Folder containing datasets
```


## Important Notes ✍️
### Model Updates:
- If you modify or retrain the ML model in Colab or elsewhere:
- Export a new .pkl file.
- Replace the old model.pkl in the project.

### Dataset Size:
Kaggle datasets are usually large — ensure your dataset fits the free hosting limits of Streamlit Cloud (~100MB total).

### Path Settings:
Ensure all file paths are relative (not absolute) for smooth deployment.

### License 📄
This project is open-source for educational and research purposes only.



<h3>Contributing </h3>
Contributions are welcome! Feel free to fork this repository, make your changes, and submit a pull request.

### Thank You for Visiting! 🙌
---
