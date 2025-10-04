from flask import Flask, render_template, request
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk

# Download VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

# Initialize Flask app
app = Flask(__name__)

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        comment = request.form['comment']
        
        # VADER Sentiment Score
        sentiment_score = sia.polarity_scores(comment)
        compound_score = sentiment_score['compound']
        
        # TextBlob Sentiment Score
        blob = TextBlob(comment)
        textblob_score = blob.sentiment.polarity
        
        # Determine sentiment category
        if compound_score >= 0.05:
            sentiment = "Positive"
        elif compound_score <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        
        # Return results dynamically
        return f"""
            <strong>Comment:</strong> {comment}<br>
            <strong>VADER Score:</strong> {compound_score:.3f}<br>
            <strong>TextBlob Score:</strong> {textblob_score:.3f}<br>
            <strong>Sentiment:</strong> {sentiment}
        """
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host = '0.0.0.0',port = 8080)
