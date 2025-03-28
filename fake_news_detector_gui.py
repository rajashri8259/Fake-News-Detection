import customtkinter as ctk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import requests
from bs4 import BeautifulSoup
import threading
import os
import time
import re
from urllib.parse import urlparse
import json
from datetime import datetime

class FakeNewsDetectorGUI:
    def __init__(self):
        self.window = ctk.CTk()
        self.window.title("Fake News Detector")
        self.window.geometry("1000x800")
        self.window.configure(fg_color="#2b2b2b")
        
        # Set theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Create main frame
        self.main_frame = ctk.CTkFrame(self.window, fg_color="transparent")
        self.main_frame.pack(padx=20, pady=20, fill="both", expand=True)
        
        # Title
        self.title_label = ctk.CTkLabel(
            self.main_frame,
            text="Fake News Detector",
            font=("Helvetica", 24, "bold")
        )
        self.title_label.pack(pady=20)
        
        # Status frame
        self.status_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.status_frame.pack(pady=10, fill="x")
        
        # Status label
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Loading model...",
            font=("Helvetica", 12)
        )
        self.status_label.pack(side="left", padx=5)
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(self.status_frame)
        self.progress_bar.pack(side="left", padx=5, fill="x", expand=True)
        self.progress_bar.set(0)
        
        # Create tabs
        self.tabview = ctk.CTkTabview(self.main_frame)
        self.tabview.pack(pady=10, padx=10, fill="both", expand=True)
        
        # Add tabs
        self.tabview.add("Article Analysis")
        self.tabview.add("Model Info")
        self.tabview.add("Fact Check")
        
        # Article Analysis Tab
        self.article_frame = ctk.CTkFrame(self.tabview.tab("Article Analysis"))
        self.article_frame.pack(padx=20, pady=20, fill="both", expand=True)
        
        # URL input
        self.url_label = ctk.CTkLabel(
            self.article_frame,
            text="Enter article URL:",
            font=("Helvetica", 14)
        )
        self.url_label.pack(pady=(20, 5))
        
        self.url_entry = ctk.CTkEntry(
            self.article_frame,
            width=600,
            placeholder_text="https://example.com/article"
        )
        self.url_entry.pack(pady=5)
        
        # Article text input
        self.text_label = ctk.CTkLabel(
            self.article_frame,
            text="Or paste article text:",
            font=("Helvetica", 14)
        )
        self.text_label.pack(pady=(20, 5))
        
        self.text_input = ctk.CTkTextbox(
            self.article_frame,
            height=200,
            width=600
        )
        self.text_input.pack(pady=5)
        
        # Analyze button
        self.analyze_button = ctk.CTkButton(
            self.article_frame,
            text="Analyze Article",
            command=self.analyze_article,
            width=200
        )
        self.analyze_button.pack(pady=20)
        
        # Result area
        self.result_frame = ctk.CTkFrame(self.article_frame)
        self.result_frame.pack(pady=20, fill="x")
        
        self.result_label = ctk.CTkLabel(
            self.result_frame,
            text="Result:",
            font=("Helvetica", 16, "bold")
        )
        self.result_label.pack(pady=10)
        
        self.result_text = ctk.CTkTextbox(
            self.result_frame,
            height=100,
            width=600
        )
        self.result_text.pack(pady=5)
        
        # Model Info Tab
        self.info_frame = ctk.CTkFrame(self.tabview.tab("Model Info"))
        self.info_frame.pack(padx=20, pady=20, fill="both", expand=True)
        
        # Model explanation
        self.explanation_text = ctk.CTkTextbox(
            self.info_frame,
            height=400,
            width=600
        )
        self.explanation_text.pack(pady=10)
        
        # Fact Check Tab
        self.fact_check_frame = ctk.CTkFrame(self.tabview.tab("Fact Check"))
        self.fact_check_frame.pack(padx=20, pady=20, fill="both", expand=True)
        
        # Fact check explanation
        self.fact_check_explanation = ctk.CTkTextbox(
            self.fact_check_frame,
            height=100,
            width=600
        )
        self.fact_check_explanation.pack(pady=10)
        self.fact_check_explanation.insert("1.0", """
This feature performs online fact-checking by:
1. Extracting key claims from the article
2. Searching fact-checking websites
3. Checking against reliable news sources
4. Verifying dates and sources

Note: This process may take a few minutes.
""")
        
        # Fact check results
        self.fact_check_results = ctk.CTkTextbox(
            self.fact_check_frame,
            height=300,
            width=600
        )
        self.fact_check_results.pack(pady=10)
        
        # Initialize model
        self.model = None
        self.vectorizer = None
        self.is_loading = True
        self.initialize_model()
        
        # Update model info immediately
        self.update_model_info()
        
    def initialize_model(self):
        """Initialize the model in a separate thread"""
        def load_model():
            try:
                # Update progress bar
                for i in range(5):
                    self.progress_bar.set(i/4)
                    time.sleep(0.5)
                
                # Load the dataset
                self.status_label.configure(text="Loading dataset...")
                df = pd.read_csv('news.csv')
                
                # Update progress
                self.progress_bar.set(0.5)
                self.status_label.configure(text="Preparing data...")
                
                # Prepare the data
                x_train, x_test, y_train, y_test = train_test_split(
                    df['text'], 
                    df['label'],
                    test_size=0.2, 
                    random_state=7
                )
                
                # Update progress
                self.progress_bar.set(0.7)
                self.status_label.configure(text="Training model...")
                
                # Create and configure vectorizer with more features
                self.vectorizer = TfidfVectorizer(
                    stop_words='english',
                    max_df=0.7,
                    ngram_range=(1, 3),  # Include phrases up to 3 words
                    max_features=10000   # Increase feature count
                )
                
                # Train the model with more iterations
                tfidf_train = self.vectorizer.fit_transform(x_train)
                self.model = PassiveAggressiveClassifier(
                    max_iter=1000,  # Increase iterations
                    C=1.0,          # Regularization parameter
                    random_state=42
                )
                self.model.fit(tfidf_train, y_train)
                
                # Calculate and store accuracy
                tfidf_test = self.vectorizer.transform(x_test)
                y_pred = self.model.predict(tfidf_test)
                self.model_accuracy = accuracy_score(y_test, y_pred)
                
                # Update progress
                self.progress_bar.set(1.0)
                self.status_label.configure(text=f"Model loaded successfully! (Accuracy: {self.model_accuracy*100:.1f}%)")
                self.is_loading = False
                
            except Exception as e:
                self.status_label.configure(text=f"Error loading model: {str(e)}")
                self.result_text.delete("1.0", "end")
                self.result_text.insert("1.0", f"Error: {str(e)}\nPlease make sure the news.csv file is present in the same directory.")
        
        # Start model loading in a separate thread
        threading.Thread(target=load_model, daemon=True).start()
        
    def fetch_article_text(self, url):
        """Fetch article text from URL"""
        try:
            self.status_label.configure(text="Fetching article...")
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Break into lines and remove leading/trailing space
            lines = (line.strip() for line in text.splitlines())
            
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            
            # Drop blank lines
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            self.status_label.configure(text="Model loaded successfully!")
            return text
            
        except Exception as e:
            self.status_label.configure(text="Error fetching article")
            return f"Error fetching article: {str(e)}"
    
    def extract_claims(self, text):
        """Extract key claims from the article text"""
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        # Filter for potential claims
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Avoid very short sentences
                # Look for sentences with numbers, dates, or strong statements
                if re.search(r'\d+', sentence) or re.search(r'\b(prove|show|confirm|verify|demonstrate|reveal)\b', sentence.lower()):
                    claims.append(sentence)
        
        return claims[:5]  # Return top 5 claims

    def check_claim(self, claim):
        """Check a single claim against fact-checking sources"""
        # List of fact-checking websites
        fact_check_sites = [
            'factcheck.org',
            'snopes.com',
            'politifact.com',
            'reuters.com/fact-check',
            'apnews.com/hub/fact-checking'
        ]
        
        results = []
        for site in fact_check_sites:
            try:
                # Search for the claim on the fact-checking site
                search_url = f"https://www.google.com/search?q=site:{site} {claim}"
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                response = requests.get(search_url, headers=headers)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract search results
                search_results = soup.find_all('div', class_='g')
                for result in search_results[:3]:  # Check top 3 results
                    title = result.find('h3')
                    if title:
                        results.append({
                            'site': site,
                            'title': title.text,
                            'url': result.find('a')['href'] if result.find('a') else None
                        })
            except Exception as e:
                print(f"Error checking {site}: {str(e)}")
        
        return results

    def verify_source(self, url):
        """Verify the credibility of the article source"""
        try:
            domain = urlparse(url).netloc
            domain = domain.lower()
            
            # List of known reliable sources
            reliable_sources = [
                'reuters.com',
                'apnews.com',
                'bbc.com',
                'nytimes.com',
                'washingtonpost.com',
                'theguardian.com',
                'aljazeera.com',
                'bloomberg.com',
                'economist.com',
                'nature.com',
                'science.org'
            ]
            
            # List of known unreliable sources
            unreliable_sources = [
                'infowars.com',
                'breitbart.com',
                'naturalnews.com',
                'beforeitsnews.com',
                'wakingtimes.com'
            ]
            
            if any(source in domain for source in reliable_sources):
                return "RELIABLE", "This source is known for reliable reporting."
            elif any(source in domain for source in unreliable_sources):
                return "UNRELIABLE", "This source has a history of publishing unreliable information."
            else:
                return "UNKNOWN", "This source is not in our database of known sources."
                
        except Exception as e:
            return "ERROR", f"Error verifying source: {str(e)}"

    def analyze_article(self):
        """Analyze the article text"""
        if self.is_loading:
            self.result_text.delete("1.0", "end")
            self.result_text.insert("1.0", "Model is still loading. Please wait...")
            return
        
        if self.model is None or self.vectorizer is None:
            self.result_text.delete("1.0", "end")
            self.result_text.insert("1.0", "Error: Model not loaded properly. Please restart the application.")
            return
        
        # Get article text
        url = self.url_entry.get()
        if url:
            article_text = self.fetch_article_text(url)
        else:
            article_text = self.text_input.get("1.0", "end-1c")
        
        if not article_text:
            self.result_text.delete("1.0", "end")
            self.result_text.insert("1.0", "Please enter either a URL or article text!")
            return
        
        try:
            # Start fact-checking in a separate thread
            def fact_check():
                self.fact_check_results.delete("1.0", "end")
                self.fact_check_results.insert("1.0", "Starting fact-check...\n\n")
                
                # Extract claims
                claims = self.extract_claims(article_text)
                self.fact_check_results.insert("end", "Key Claims Found:\n")
                for i, claim in enumerate(claims, 1):
                    self.fact_check_results.insert("end", f"{i}. {claim}\n")
                
                # Verify source if URL is provided
                if url:
                    self.fact_check_results.insert("end", "\nSource Verification:\n")
                    reliability, message = self.verify_source(url)
                    self.fact_check_results.insert("end", f"Status: {reliability}\n{message}\n")
                
                # Check claims
                self.fact_check_results.insert("end", "\nFact-Checking Results:\n")
                for claim in claims:
                    self.fact_check_results.insert("end", f"\nChecking claim: {claim}\n")
                    results = self.check_claim(claim)
                    if results:
                        self.fact_check_results.insert("end", "Related fact-checks found:\n")
                        for result in results:
                            self.fact_check_results.insert("end", f"- {result['title']}\n")
                            if result['url']:
                                self.fact_check_results.insert("end", f"  URL: {result['url']}\n")
                    else:
                        self.fact_check_results.insert("end", "No fact-checks found for this claim.\n")
                
                self.fact_check_results.insert("end", "\nFact-checking complete!")
            
            # Start fact-checking thread
            threading.Thread(target=fact_check, daemon=True).start()
            
            # Continue with ML prediction
            self.status_label.configure(text="Analyzing article...")
            tfidf_article = self.vectorizer.transform([article_text])
            prediction = self.model.predict(tfidf_article)
            
            # Get decision function score
            decision_score = self.model.decision_function(tfidf_article)[0]
            confidence = min(abs(decision_score) * 100, 100)
            
            # Get feature importance
            feature_names = self.vectorizer.get_feature_names_out()
            coef = self.model.coef_[0]
            top_features = sorted(zip(coef, feature_names), reverse=True)[:5]
            
            # Display result with more details
            result_text = f"""
Article Analysis Result:
------------------------
Prediction: {'REAL' if prediction[0] == 'REAL' else 'FAKE'}
Confidence: {confidence:.2f}%
Model Accuracy: {self.model_accuracy*100:.1f}%

Key Features Affecting Prediction:
"""
            for coef, feature in top_features:
                result_text += f"- {feature}: {coef:.2f}\n"
            
            result_text += """
Note: This is an AI-based prediction and should be used as one of many tools for fact-checking.
Check the Fact Check tab for detailed online verification.
"""
            self.result_text.delete("1.0", "end")
            self.result_text.insert("1.0", result_text)
            self.status_label.configure(text="Analysis complete!")
            
        except Exception as e:
            self.result_text.delete("1.0", "end")
            self.result_text.insert("1.0", f"Error analyzing article: {str(e)}")
            self.status_label.configure(text="Error during analysis")
    
    def update_model_info(self):
        """Update the model information display"""
        info_text = """
How the Fake News Detector Works:

1. Data Processing:
   - The model uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numerical features
   - Stop words are removed to focus on meaningful content
   - Words and phrases (up to 3 words) are analyzed
   - Features are weighted based on their importance in the document

2. Model Architecture:
   - Uses a Passive Aggressive Classifier with 1000 training iterations
   - Trained on a dataset of labeled news articles
   - Learns to distinguish between real and fake news based on linguistic patterns
   - Includes regularization to prevent overfitting

3. Prediction Process:
   - Input text is converted to TF-IDF features
   - Model predicts whether the article is REAL or FAKE
   - Provides confidence score based on decision function
   - Shows key features that influenced the prediction

4. Model Performance:
   - Shows overall model accuracy on test data
   - Displays top features affecting each prediction
   - Confidence score indicates prediction strength

5. Limitations:
   - Model accuracy depends on training data quality
   - May not catch sophisticated fake news
   - Should be used as one of many fact-checking tools
   - Performance may vary with different types of news

6. Usage Instructions:
   - Enter a news article URL or paste the article text
   - Click "Analyze Article" to get the prediction
   - The model will show if the article is likely REAL or FAKE
   - Review the key features that influenced the prediction

Note: This tool is for educational purposes only and should not be the sole source of truth verification.
"""
        self.explanation_text.delete("1.0", "end")
        self.explanation_text.insert("1.0", info_text)
    
    def run(self):
        """Run the application"""
        self.window.mainloop()

if __name__ == "__main__":
    app = FakeNewsDetectorGUI()
    app.run() 