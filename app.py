from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

app = Flask(__name__)
sia = SentimentIntensityAnalyzer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        text = request.form['text']
        scores = sia.polarity_scores(text)
        
        # Visualization
        labels = ['Negative', 'Neutral', 'Positive']
        sizes = [scores['neg'], scores['neu'], scores['pos']]
        colors = ['#ff9999','#66b3ff','#99ff99']
        explode = (0.1, 0, 0)  # explode the 1st slice (Negative)

        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.savefig('static/sentiment_pie.png')
        plt.close()

        return render_template('result.html', scores=scores, text=text)

if __name__ == '__main__':
    app.run(debug=True)
