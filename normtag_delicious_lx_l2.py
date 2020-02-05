# use normtag
# original data
# data format: userID bookmarkID tagID timesatamp
import random
import math
import operator
import pandas as pd

file_path = "./user_taggedbookmarks-timestamps.dat"
# dict records {userid:{items1:[tag1,tag2],...}}
records = {}
train_data = dict()
test_data = dict()

user_tags = dict()
tag_items = dict()
user_items = dict()
item_tags = dict()
tag_users = dict()
item_users = dict()

# show user top N
def recommend(user,N):
    recommend_items = dict()
    # score item
    tagged_items = user_items[user]
    for tag,wut in user_tags[user].items():
        for item, wti in tag_items[tag].items():
            if item in tagged_items:
                continue
            # NormTagBased-1算法
            #norm = len(tag_users[tag].items()) * len(user_tags[user].items())
            # NormTagBased-2算法
            #norm = len(tag_items[tag].items()) * len(user_tags[user].items())
            # tagebased-TFIDF algorithm
            norm = math.log(len(tag_users[tag].items()) + 1)

            if item not in recommend_items:
                recommend_items[item] = wut*wti/norm
            else:
                recommend_items[item] = recommend_items[item] + wut*wti/norm
    return sorted(recommend_items.items(), key=operator.itemgetter(1), reverse=True)[0:N]

# compute recall and precision with test_data
def precisionAndRecall(N):
    hit = 0
    h_recall = 0
    h_precision = 0
    for user, items in test_data.items():
        if user not in train_data:
            continue
        # recommend top n of user
        rank = recommend(user,N)
        for item, rui in rank:
            if item in items:
                hit = hit + 1
             #print('user,items,rui is', user,item,rui)
        h_recall = h_recall + len(items)
        h_precision = h_precision + N
    #print('一共命中 %d 个, 一共推荐 %d 个, 用户设置tag总数 %d 个' %(hit, h_precision, h_recall))
    # retrun recall and precision
    return (hit/(h_precision*1.0)), (hit/(h_recall*1.0))

#assume the result using the test_data
def testRecommand():
    print("推荐结果评估")
    print("%3s %10s %10s" % ('N',"精确率",'召回率'))
    for n in [5,40,100]:
        precision,recall = precisionAndRecall(n)
        print("%3d %10.3f%% %10.3f%%" % (n, precision * 100, recall * 100))



#loading data
def load_data():
    print('loading data...')
    df = pd.read_csv(file_path, sep ='\t')
    for i in range(len(df)):
        uid = df['userID'][i]
        iid = df['bookmarkID'][i]
        tag = df['tagID'][i]
        # if key not exist,set default{}
        records.setdefault(uid,{})
        records[uid].setdefault(iid,[])
        records[uid][iid].append(tag)
    print("数据集大小为 %d." % (len(df)))
    print("设置tag的人数 %d." % (len(records)))
    print("数据加载完成\n")
    #df = df.head(10)
    #print('df head 10:', df)

def train_test_split(ratio, seed=100):
    random.seed(seed)
    #a = random.random()
    #print('rrandom.random is:', a)
    for u in records.keys():
        for i in records[u].keys():
            if random.random()<ratio:
                test_data.setdefault(u,{})
                test_data[u].setdefault(i,[])
                for t in records[u][i]:
                    test_data[u][i].append(t)
            else:
                train_data.setdefault(u,{})
                train_data[u].setdefault(i,[])
                for t in records[u][i]:
                    train_data[u][i].append(t)
    print("训练集样本数 %d, 测试集样本数 %d" % (len(train_data),len(test_data)))

#set mat[index, item] = 1
def addValueToMat(mat, index, item, value=1):
    if index not in mat:
        mat.setdefault(index,{})
        mat[index].setdefault(item, value)
    else:
        if item not in mat[index]:
            mat[index][item] = value
        else:
            mat[index][item] += value

def initStat():
    records = train_data
    for u, items in records.items():
        for i, tags in items.items():
            for tag in tags:
                addValueToMat(user_tags, u, tag, 1)
                addValueToMat(tag_items, tag, i, 1)
                addValueToMat(user_items, u, i, 1)
                addValueToMat(item_tags, i, tag, 1)
                addValueToMat(tag_users, tag, u, 1)
                addValueToMat(item_users, i, u, 1)
    print("user_tags, tag_items, user_items初始化完成.")
    print("user_tags大小 %d, tag_items大小 %d, user_items大小 %d" % (len(user_tags), len(tag_items), len(user_items)))

# prrint dict with top n
def pri_dict(mat, n):
    new_a = {}
    for i,(k,v) in enumerate(mat.items()):
        new_a[k] = v
        if i==n:
            print("top n of mat is ", new_a)
            break

    
load_data()
train_test_split(0.2)
initStat()
#pri_dict(user_tags, 2)
#pri_dict(user_items, 2)
#pri_dict(item_tags, 2)
#pri_dict(item_users, 2)
#pri_dict(tag_items, 2)
#pri_dict(tag_users, 2)
testRecommand()