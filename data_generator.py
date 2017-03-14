import numpy as np
import pandas as pd

tri = pd.read_csv('data/training_info.csv')
trs = pd.read_csv('data/training_set.csv')

ti = pd.read_csv('data/test_info.csv')
ts = pd.read_csv('data/test_set.csv')

n = len(tri)
send = np.empty(n, dtype='S40')
corr = {}
for a in trs.values:
    for b in a[1].split():
        corr[int(b)] = a[0]

vals = tri.values
for i in range(n):
    send[i] = corr[vals[i][0]]
    
tri.insert(0, 'sender', send)


n = len(ti)
send = np.empty(n, dtype='S40')
corr = {}
for a in ts.values:
    for b in a[1].split():
        corr[int(b)] = a[0]

vals = ti.values
for i in range(n):
    send[i] = corr[vals[i][0]]
    
ti.insert(0, 'sender', send)


address_ids = dict(zip(list(set(ti['sender'].values)), range(125)))

vals = ti.values
idc = np.zeros(len(ti), dtype=int)
for i in range(len(ti)):
    idc[i] = address_ids[vals[i][0]]
    
vals = tri.values
idc2 = np.zeros(len(tri), dtype=int)
for i in range(len(tri)):
     idc2[i] = address_ids[vals[i][0]]
        
tri.insert(1, 'sender_id', idc2)
ti.insert(1, 'sender_id', idc)

cur = 125
for x in tri.values:
    s = x[5]
    for e in s.split():
        if not e in address_ids and '@' in e:
            address_ids[e] = cur
            cur += 1
            
idc = []
recipients = []
vals = tri.values
for i in range(len(tri)):
    r = []
    for e in vals[i][5].split():
        if '@' in e:
            r.append(address_ids[e])
    idc.append(r)
    
tri.insert(5, 'recipient_id', pd.Series(idc, dtype=list))


vals = tri.values
for i in range(len(vals)):
    r = []
    for s in vals[i][6].split():
        if '@' in s:
            r.append(s)
    vals[i][6] = ' '.join(r)
    
tri['recipients'] = vals[:, 6]


tri.to_csv('data/train.csv', index=False)
ti.to_csv('data/test.csv', index=False)