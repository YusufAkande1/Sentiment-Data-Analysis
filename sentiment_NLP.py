import pandas as pd
import re
from textblob import TextBlob

df = pd.read_csv("sentiment_dataset.csv")


# =========================================
# 8. TEXT PREPROCESSING (NLP)
# =========================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

df['Clean_Text'] = df['Text'].apply(clean_text)

print("\nText Cleaning Completed!")


# =========================================
# 9. SENTIMENT ANALYSIS USING TEXTBLOB
# =========================================

df['Polarity'] = df['Clean_Text'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Convert polarity to sentiment
def get_sentiment(score):
    if score > 0:
        return 'Positive'
    elif score < 0:
        return 'Negative'
    else:
        return 'Neutral'

df['Predicted_Sentiment'] = df['Polarity'].apply(get_sentiment)

print("\nSentiment Prediction Completed!")
