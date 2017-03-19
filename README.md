# Altegrad_Challenge

### Our repo for the March 2017 Altegrad Challenge with students from the MVA and the Saclay Data Science Master.

#### https://inclass.kaggle.com/c/master-data-science-mva-data-competition-2017

This is a supervised task consisting of predicting the recipients of emails from a subset of the Enron corpus. For each mail you have the Sender, a Timestamp and the mail Body. It's not allowed to get more data.

Our approached is based on this article:

http://www.cs.cmu.edu/~wcohen/postscript/cc-predict-submitted.pdf

And we use the "memory model" idea from this article:

http://ieeexplore.ieee.org/document/6273570/

In short, our method consists in:

- Predicting the mails from each sender separately.
- Use tf-idf on the whole corpus with no stemming, and tokenizing with spaces only.
- Using the K nearest neigbhors of each mail in the tf-idf corpus as candidate recipients.
- Use additionnal features (using the timestamp) and train a classifier to discriminate between good and bad recipients.

### Explanation of the features:

- 1: Number of candidate mails received by the candidate. The more, the more probable he's going to receive other mails.
- 2: Sum of the cosine similarities of the mails received by this candidate.
- 3: Does the name or surname of this candidate appear in the first 30 symbols of the mail ?
- 4: Sum of the difference of timestamp of received candidate mail and the current mail.
- 5: Total number of mails reveived by this candidate (not only in the candidate mails).
- 6: Sum of similarities*(difference of timestamp between the new mail and the candidate mail). The purpose of this feature is to give more importance to more recent mails that have high cosine similariy.
- 7: Sum of the diffenrences of timestamp to the power $-\lambda$ over all the mails of the candidate recipient. This the memory model feature we took from this article: http://ieeexplore.ieee.org/document/6273570/

You can run our approach by running two python scripts :

data_generator.py

predictor.py
