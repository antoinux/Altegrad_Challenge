# Altegrad_Challenge

### Our repo for the March 2017 Altegrad Challenge with students from the MVA and the Saclay Data Science Master.

#### https://inclass.kaggle.com/c/master-data-science-mva-data-competition-2017

This is a supervised task consisting of predicting the recipients of emails from a subset of the Enron corpus. For each mail you have the Sender, a Timestamp and the mail Body. It's not allowed to get more data.

Our approached is based on this article:

http://www.cs.cmu.edu/~wcohen/postscript/cc-predict-submitted.pdf

And we use the "memory model" idea from this article:

http://ieeexplore.ieee.org/document/6273570/

In short, our method consists in:

- Predicting the mails of each sender separately
- Use tf-idf on the whole corpus with no stemming, and tokenizing with spaces only.
- Using the K nearest neigbhors of each mail in the tf-idf corpus as candidate recipients.
- Use additionnal features (using the timestamp) and train a classifier to discriminate between good and bad recipients.


You can run our approach by running two python scripts :

data_generator.py
predictor.py