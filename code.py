Import pandas as pd
From sklearn.model_selection import train_test_split
From sklearn.feature_extraction.text import TfidfVectorizer
From sklearn.naive_bayes import MultinomialNB
From sklearn.metrics import accuracy_score, classification_report
Import matplotlib.pyplot as plt
# Load and preprocess data
Data = pd.read_csv(‘customer_reviews.csv’)  # Make sure the file path is correct
X = data[‘text’]
Y = data[‘sentiment’]
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Convert text to TF-IDF vectors
Tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
# Train a Naïve Bayes classifier
Clf = MultinomialNB()
Clf.fit(X_train_tfidf, y_train)
# Make predictions
Y_pred = clf.predict(X_test_tfidf)
# Evaluate the model
Accuracy = accuracy_score(y_test, y_pred)
Print(f’Accuracy: {accuracy}’)
Print(classification_report(y_test, y_pred))
# Create a bar chart to visualize the distribution of sentiment labels
Sentiment_counts = y.value_counts()
Sentiment_labels = sentiment_counts.index
Plt.bar(sentiment_labels, sentiment_counts)
Plt.xlabel(‘Sentiment’)
Plt.ylabel(‘Count’)
Plt.title(‘Sentiment Distribution in Customer Reviews’)
Plt.show()