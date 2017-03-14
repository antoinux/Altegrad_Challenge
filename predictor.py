import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
import scipy

print("Loading Data")

df = pd.read_csv('data/train.csv').sort_values('date')
df['recipient_id'] = df['recipient_id'].apply(literal_eval)
X, X_time, y = df[['sender_id', 'body', 'date']].values, df['date'].values, df['recipient_id'].values

df_test = pd.read_csv('data/test.csv').sort_values('date')
X2 = df_test[['sender_id', 'body', 'date', 'mid']].values

# Removing digits from the corpus
for i in range(len(X)):
    X[i][1] = ''.join([d for d in X[i][1] if not d.isdigit()])
    
for i in range(len(X2)):
    X2[i][1] = ''.join([d for d in X2[i][1] if not d.isdigit()])
    

print("Spliting data among senders")
    
X_senders = [[] for i in range(125)]
y_senders = [[] for i in range(125)]
X_senders2 = [[] for i in range(125)]
test_mids = [[] for i in range(125)]
X_time = [[] for i in range(125)]
X_time2 = [[] for i in range(125)]
X_train_text = [[] for i in range(125)]
X_test_text = [[] for i in range(125)]

vect = TfidfVectorizer(min_df=10)
X_body_train = vect.fit_transform(X[:, 1])
X_body_test = vect.transform(X2[:, 1])

norm1 = scipy.sparse.linalg.norm(X_body_train, axis=1)
for i in range(len(norm1)):
    if norm1[i] == 0.:
        norm1[i] = 1.
X_body_train = X_body_train.multiply(scipy.sparse.csr_matrix(1./norm1.reshape(-1, 1)))

norm2 = scipy.sparse.linalg.norm(X_body_test, axis=1)
for i in range(len(norm2)):
    if norm2[i] == 0.:
        norm2[i] = 1.
X_body_test = X_body_test.multiply(scipy.sparse.csr_matrix(1./norm2.reshape(-1, 1)))

for i in range(len(X)):
    X_senders[X[i][0]].append(i)
    X_time[X[i][0]].append(np.datetime64(X[i][2]))
    y_senders[X[i][0]].append(y[i])
    X_train_text[X[i][0]].append(X[i][1])
    
for i in range(len(X2)):
    X_senders2[X2[i][0]].append(i)
    X_time2[X2[i][0]].append(np.datetime64(X2[i][2]))
    test_mids[X2[i][0]].append(X2[i][3])
    X_test_text[X2[i][0]].append(X2[i][1])
    
y_train, X_train_tf, X_test_tf= [], [], []
X_train_time, X_test_time = [], []
for s in range(125):
    X_senders[s] = X_body_train[np.array(X_senders[s]), :]
    X_senders2[s] = X_body_test[np.array(X_senders2[s]), :]
    X_time[s] = np.array(X_time[s]).astype('int64')
    X_time2[s] = np.array(X_time2[s]).astype('int64')
    
    X_train_time.append(X_time[s])
    X_test_time.append(X_time2[s])
    X_train_tf.append(X_senders[s])
    X_test_tf.append(X_senders2[s])
    
    y_train.append(y_senders[s])
    
recipient_ids = {}
for l in df[['recipient_id', 'recipients']].values:
    a = l[1].split()
    for i in range(len(a)):
        recipient_ids[l[0][i]] = a[i]
    
n_people = max(recipient_ids.keys())+1
recipient_names = []
for i in range(n_people):
    if i not in recipient_ids:
        recipient_names.append([])
        continue
    address = recipient_ids[i]
    s = address.split('@')[0].split('.')
    if len(s) == 2:
        recipient_names.append(s)
    else:
        recipient_names.append([])
        
        
def name_in_header(name, mail):
    mail = mail[:30].lower()
    for s in name:
        if s.lower() in mail:
            return 1.
    return 0.
 
# Computing how many mails each sender sent to each recipient.
sent_by_sender = np.zeros((125, n_people))
for s in range(125):
    for l in y_train[s]:
        for r in l:
            sent_by_sender[s][r] += 1
s = np.sum(sent_by_sender, axis=1)
for i in range(len(s)):
    if s[i] == 0.:
        s[i] = 1.
sent_by_sender /= np.sum(sent_by_sender, axis=1).reshape(-1, 1)

baseline = np.argsort(sent_by_sender)[:, ::-1][:, :10]

n_neigbhors = 40
l = 6.5
X_train_full, y_train_full = [], []

mail_times = [[[] for j in range(n_people)] for i in range(125)]
for s in range(125):
    for j in range(len(X_train_time[s])):
        for r in y_train[s][j]:
            mail_times[s][r].append(X_train_time[s][j])

print("Preprocessing - Building data sets")

for s in range(125):
    # We can process the similiraties like this because the tf-idf vectors were already normalized.
    cosine_similarities_matrix = X_train_tf[s].dot(X_train_tf[s].transpose())
    
    X_train, y_train_true = np.empty((0, 7)), np.empty(0)
    
    for j in range(0, len(X_train_text[s])):
        cosine_similarities = np.array(cosine_similarities_matrix[j].todense())[0]
            
        # don't forget to not take the first one, which is the mail itself and will always have a similarity equal to 1.
        closests = np.argsort(cosine_similarities)[::-1][1:n_neigbhors+1]
        candidates_local_keys = {}
        cur = 0
        for m in closests:
            for r in y_train[s][m]:
                if r not in candidates_local_keys:
                    candidates_local_keys[r] = cur
                    cur += 1
        n_candidates = len(candidates_local_keys)
        
        # That's where we compute the 7 features for the classifier. For a description of these, check the README.
        
        features = np.zeros((n_candidates, 7))
        for m in closests:
            for r in y_train[s][m]:
                if r in candidates_local_keys:
                    features[candidates_local_keys[r]][0] += 1.
                    features[candidates_local_keys[r]][1] += cosine_similarities[m]
                    features[candidates_local_keys[r]][3] += X_train_time[s][j] - X_train_time[s][m]
                    features[candidates_local_keys[r]][5] += cosine_similarities[m]*(X_train_time[s][j] - X_train_time[s][m])
        
        for r in candidates_local_keys:
                features[candidates_local_keys[r]][2] = name_in_header(recipient_names[r], X_train_text[s][j])
                features[candidates_local_keys[r]][4] = sent_by_sender[s][r]
                    
                a = np.array(mail_times[s][r])
                features[candidates_local_keys[r]][6] = ((X_train_time[s][j] - a[a < X_train_time[s][j]])**(-l)).sum()
                if len(mail_times[s][r]) != 0:
                    features[candidates_local_keys[r]][6] /= len(mail_times[s][r])
                
        for i in range(n_candidates):
            if features[i][0] == 0.:
                features[i][0] = 1.
        
        features[:, 1] /= features[:, 0]
        features[:, 3] /= features[:, 0]
        features[:, 0] /= len(closests)
        
        X_train = np.vstack((X_train, features))
        
        y_mail = np.zeros(n_candidates)
        for r in candidates_local_keys:
            y_mail[candidates_local_keys[r]] = 1. if r in y_train[s][j] else 0.
        
        y_train_true = np.hstack((y_train_true, y_mail))
    
    X_train_full.append(X_train)
    y_train_full.append(y_train_true)
    
    
l = 6.5
X_test_full, starting_ids, keys_to_rs = [], [], []

for s in range(125):
    cosine_similarities_matrix = X_test_tf[s].dot(X_train_tf[s].transpose())
    y_pred_sender = np.empty((len(X_test_text[s]), 10))
    
    X_test_sender = np.empty((0, 7))
    sender_starting_ids = []
    sender_keys_to_r = []

    for j in range(len(X_test_text[s])):
        sender_starting_ids.append(len(X_test_sender))
        cosine_similarities = np.array(cosine_similarities_matrix[j].todense())[0]

        closests = np.argsort(cosine_similarities)[::-1][:n_neigbhors]

        candidates_local_keys = {}
        cur = 0
        for m in closests:
            for r in y_train[s][m]:
                if r not in candidates_local_keys:
                    candidates_local_keys[r] = cur
                    cur += 1
        n_candidates = len(candidates_local_keys)

        features = np.zeros((n_candidates, 7))
        for m in closests:
            for r in y_train[s][m]:
                if r in candidates_local_keys:
                    features[candidates_local_keys[r]][0] += 1.
                    features[candidates_local_keys[r]][1] += cosine_similarities[m]
                    features[candidates_local_keys[r]][3] += X_test_time[s][j] - X_train_time[s][m]
                    features[candidates_local_keys[r]][5] += cosine_similarities[m]*(X_test_time[s][j] - X_train_time[s][m])

        for r in candidates_local_keys:
                features[candidates_local_keys[r]][2] = name_in_header(recipient_names[r], X_test_text[s][j])
                features[candidates_local_keys[r]][4] = sent_by_sender[s][r]
                
                a = np.array(mail_times[s][r])
                
                features[candidates_local_keys[r]][6] = ((X_test_time[s][j] - a[a < X_test_time[s][j]])**(-l)).sum()
                if len(mail_times[s][r]) != 0:
                    features[candidates_local_keys[r]][6] /= len(mail_times[s][r])

        for i in range(n_candidates):
            if features[i][0] == 0.:
                features[i][0] = 1.

        features[:, 1] /= features[:, 0]
        features[:, 3] /= features[:, 0]
        features[:, 0] /= len(closests)

        X_test_sender = np.vstack((X_test_sender, features))
        
        keys_to_r = {candidates_local_keys[r]:r for r in candidates_local_keys}
        sender_keys_to_r.append(keys_to_r)
    
    X_test_full.append(X_test_sender)
    sender_starting_ids.append(len(X_test_sender))
    starting_ids.append(sender_starting_ids)
    keys_to_rs.append(sender_keys_to_r)
    
    

print("Training models and predicting on test")

y_pred = []
models = []
max_leaf_nodes = 45

n_estimators = 10 # sub was made with n_estimators=1000

for s in range(125):
    clf = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, n_jobs=-1).fit(X_train_full[s], y_train_full[s])
    models.append(clf)
    y_pred_sender = np.empty((len(X_test_text[s]), 10))
        
    if len(X_test_text[s]) == 0:
            pass
    elif clf.n_classes_ == 1:
        y_pred_sender = np.array([baseline[s] for i in range(len(X_test_text[s]))])
    else:
        
        raw_pred = clf.predict_proba(X_test_full[s])[:, 1]
            
        for j in range(len(X_test_text[s])):
            pre = list(raw_pred[starting_ids[s][j]:starting_ids[s][j+1]].argsort()[::-1][:10])
            for i in range(len(pre)):
                pre[i] = keys_to_rs[s][j][pre[i]]

            # if we don't have enough candidates, fill with baseline
            cur = 0
            while len(pre) < 10:
                if baseline[s][cur] not in keys_to_rs[s][j].values():
                    pre.append(baseline[s][cur])
                cur += 1
            y_pred_sender[j] = np.array(pre)
    y_pred.append(y_pred_sender)
        
with open('data/your_sub.txt', 'w') as f:
    f.write('mid,recipients\n')
    for s in range(125):
        for i in range(len(y_pred[s])):
            f.write('{},'.format(test_mids[s][i]))
            for r in y_pred[s][i]:
                f.write(recipient_ids[r] + ' ')
            f.write('\n')