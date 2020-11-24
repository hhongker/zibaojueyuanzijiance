from sklearn.cluster import KMeans
with open("train.txt",'r') as f:
    datas=f.readlines()


res = []
for i in datas:
    print(i)
    j = i.split(" ")
    if len(j)>=2:
        for k in j[1:]:
            q = k.split(",")
            q1 = int(q[2]) - int(q[0])
            q2 = int(q[3]) - int(q[1])
            res.append([q2,q1])


model_rows=KMeans(n_clusters=9).fit(res) 
model_rows.cluster_centers_.round(decimals=4)