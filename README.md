# Sentiment-Analysis-Financial-Tweets

In the given notebooks, a sentiment analysis of financial news tweets is done, based on the Kaggle dataset [Sentiment Analysis for Financial Tweets](https://www.kaggle.com/ankurzing/sentiment-analysis-for-financial-news).

Out of the more than 4,800 tweets, only a small fraction was used to explore the performance of the methodology for small and unbalanced data sets.
More precisely, 175 positive, 70 neutral and 35 negative tweets were used. 

The procedure is described in the notebook *NLP Sentiment Analysis Final*. The final model is a support vector classifier with regularization parameter `C=9` based on a Word2Vec representation of the data. The classifier and the mean vector (needed to calculate the word2vec representation) are stored in the files *final_model.joblib* and *mean_vec.joblib*. 

The final model's performance is evaluated based on the remaining 4,500 tweets in the notebook *Interpretation of Results*. The evaluation suggests that the model's precision is poor (around 50%). As only positive and negative tweets might be of interest, neutral tweets could be considered as noise. Thus, in a second try, the same methodology as before was employed with disregarded neutral tweets (cf. notebook *NLP Sentiment Analysis Pos-Neg*). The model trained on positive and negative data only has a precision of approx. 72.6%. The latter model is classifier based on logistic regression with inverse of the regularization parameter `C=0.6` and a TF-IDF representation of the data. The final model and vectorizer are stored in the files *final_model_pos_neg.joblib* and *vectorizer_pos_neg.joblib*. 
