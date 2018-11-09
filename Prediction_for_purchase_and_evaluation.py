# -*- coding:utf-8 -*-
#sklearn model默认是float32 是float64？如果number超过一定长度 就不能作为可接受的input
import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold,GridSearchCV
import sklearn
from tabulate import tabulate
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler

df0=pd.read_csv('./dataset/combined/dt4.csv')
df1=pd.read_csv("./dataset/combined/dt1.csv")
df2=pd.read_csv("./dataset/combined/dt2.csv")
df3=pd.read_csv('./dataset/combined/dt3-1.csv')

y=df1['if_purchased'] #目标为y
y = np.array(y)

ft1=[]
ft2=[]
un=['Unnamed: 0','Unnamed: 0.1','Unnamed: 0.1.1','Unnamed: 0.1.1.1','Unnamed: 0.1.1.1.1','Unnamed: 0.1.1.1.1.1','Unnamed: 0.1.1.1.1.1.1']
#去掉不相关并且文本型字符串 已经去掉了所有非数值型
x0=df0.drop(un+['user_id','book_id','author_id','if_purchased'],axis=1)
x1=df1.drop(un+['user_id','book_id','if_purchased'],axis=1)
x2=df2.drop(un+['user_id','book_id','author_id','if_purchased'],axis=1)
x3=df3.drop(un+['user_id','book_id','author_id','if_purchased']+['Unnamed: 0.1.1.1.1.1.1.1','Unnamed: 0.1.1.1.1.1.1.1.1','Unnamed: 0.1.1.1.1.1.1.1.1.1'],axis=1)


X=[x0,x1,x2,x3]
# print("here is the shape")
# for eachx in X:
#     #将所有被认为string的number转化回float
#     print(eachx.dtypes)
    # eachx[eachx.columns] = eachx[eachx.columns].apply(pd.to_numeric, errors='coerce',downcast='float')
    # print("after transform")
    # print(eachx.dtypes)

#     # scaler = MinMaxScaler()
#     # print(scaler.fit(eachx))
#     # print(scaler.transform(eachx))
#
#     # imp = Imputer(missing_values='NAN', strategy='most_frequent',axis=0)
#     # # 上面'NAN'表示无效值在数据文件中的标识是'NAN',strategy='mean'表示用全局平均值代替无效值，axis=0表示对列进行处理
#     # imp.fit(eachx)#训练一个缺失值处理模型
#     # eachx = imp.transform(eachx)
#     print(eachx.shape)

split=5
def average(x):
    if type(x) is str:
        pass
    else:
        return x/split

def clf(x,y,index,cols):

    cnt = 0
    acc_mat =pd.DataFrame(np.zeros(shape=(6, split)), dtype=float)
    auc_mat = pd.DataFrame(np.zeros(shape=(6, split)), dtype=float)
    f1_mat = pd.DataFrame(np.zeros(shape=(6, split)), dtype=float)
    clf_namelist = ['Naive Beyasian', 'Decision Tree', 'Random Forest', 'SVM', 'Ada Boost', 'Gradient Boosting']
    columns = ["classifier", 'precision_score_for_pos', 'recall_score_for_pos', 'f1_score_for_pos', \
               'precision_score_for_neg', 'recall_score_for_neg', 'f1_score_for_neg', \
               'accuracy_for_pos', 'accuacy_for_neg', \
               'auc_for_pos', 'auc_for_neg', \
               'weighted_aver_precision', 'weighted_aver_recall','weighted_aver_f1_score', \
               'weighted_aver_accuracy','weighted_aver_auc']

    pd_eval = pd.DataFrame(np.zeros(shape=(6, 11+5)), columns=columns, dtype=float)
    #print(pd_eval)

    kf = KFold(n_splits=split, shuffle=True, random_state=1)
    k=0
    for train_index, test_index in kf.split(x):
        cnt = cnt+ 1
        # print('train_index', train_index, 'test_index', test_index)
        x_train, y_train = x[train_index], y[train_index]
        x_test, y_test = x[test_index], y[test_index]
        print(x_train.shape)
        print(x_test.shape)
    #x_train, x_test, y_train, y_test = train_test_split(
        #x, y, test_size=0.2, random_state=1)
        print("start nb")
        from sklearn.naive_bayes import GaussianNB
        clf0 = GaussianNB()
        clf0.fit(x_train,y_train)

        print("start dt")
        from sklearn import tree #sklearn包依赖scipy包 故都需要使用pip进行加载
        clf1 = tree.DecisionTreeClassifier(criterion='gini',max_depth=6)
        clf1= clf1.fit(x_train,y_train)
        print("acc is")
        print(accuracy_score(y_test, clf1.predict(x_test)))

        #param_test2 = {'max_depth': [3,5,8], \
                    #   'min_samples_split': [3,5,8], \
                    #   'n_estimators': [100,500,1000,1500]
                    #   }
        #gsearch2 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=60,
                                   #                              min_samples_leaf=20, max_features='sqrt',
                              #                                   oob_score=True, random_state=10),
                             #   param_grid=param_test2, scoring='accuracy', iid=False)
        #gsearch2.fit(x_test, y_test)
        #clf2 = gsearch2.best_estimator_
        #print(gsearch2.best_estimator_)
        #print(gsearch2.cv_results_)
        #print(gsearch2.best_score_)

        print("start rf")
        from sklearn.ensemble import RandomForestClassifier
        clf2 = RandomForestClassifier(n_estimators=100)
        clf2 = clf2.fit(x_train, y_train)
        print("acc is")
        print(accuracy_score(y_test,clf2.predict(x_test)))
        headers = ["name", "score"]
        values = sorted(zip(cols, clf2.feature_importances_), key=lambda x: x[1] * -1)
        print("this is feature imp ranking:")
        print(values)
        # sum1=0
        # sum2=0
        # sum3=0
        # for eachtuple in values:
        #     print(eachtuple[0]+',',eachtuple[1])
        #     if eachtuple[0] in ft1:
        #         sum1+=eachtuple[1]
        #     else:
        #         if eachtuple[0] in ft2:
        #             sum2+=eachtuple[1]
        #         else:
        #             sum3+=eachtuple[1]
        # print(sum1,sum2,sum3)


        # Print the feature ranking
        print("start knn")
        from sklearn import neighbors
        clf3 = neighbors.KNeighborsClassifier(n_neighbors=5)
        # from sklearn.svm import SVC
        # clf3 = SVC()
        #knn is really slow when data is getting bigger
        clf3.fit(x_train, y_train)
        print("acc is")
        print(accuracy_score(y_test, clf3.predict(x_test)))

        print("start ada boost")
        from sklearn.ensemble import AdaBoostClassifier
        clf4 = AdaBoostClassifier(n_estimators=100)
        clf4 = clf4.fit(x_train, y_train)

        print("start gbdt")
        from sklearn.ensemble import GradientBoostingClassifier
        clf5 = GradientBoostingClassifier(learning_rate=0.01, n_estimators=100)
        clf5 = clf5.fit(x_train, y_train)
        print("acc is")
        print(accuracy_score(y_test, clf5.predict(x_test)))

        clf_list = [clf0, clf1, clf2, clf3, clf4, clf5]

        for i in range(len(clf_list)):
            pd_eval[columns[0]][i] = clf_namelist[i]
            pd_eval[columns[1]][i] += precision_score(y_test,clf_list[i].predict(x_test),pos_label=1)
            pd_eval[columns[2]][i] += recall_score(y_test, clf_list[i].predict(x_test),pos_label=1)
            pd_eval[columns[3]][i] += f1_score(y_test, clf_list[i].predict(x_test),pos_label=1)


            pd_eval[columns[4]][i] += precision_score(y_test, clf_list[i].predict(x_test),pos_label=0)
            pd_eval[columns[5]][i] += recall_score(y_test, clf_list[i].predict(x_test),pos_label=0)
            pd_eval[columns[6]][i] += f1_score(y_test, clf_list[i].predict(x_test),pos_label=0)

            f1_mat[k][i] = (0.5)* f1_score(y_test, clf_list[i].predict(x_test),pos_label=1) + \
                                           (0.5)*f1_score(y_test, clf_list[i].predict(x_test),pos_label=0)


            predict = clf_list[i].predict(x_test)

            matrix = confusion_matrix(y_test,predict)
            #print(matrix)
            tn=matrix[0][0]
            fn=matrix[1][0]
            fp=matrix[0][1]
            tp=matrix[1][1]

            pd_eval[columns[7]][i] += tp/(fn+tp) #acc for pos
            pd_eval[columns[8]][i] += tn/(tn+fp) #acc for neg

            acc_mat[k][i] = (tp+tn)/(tp+tn+fp+fn)

            predict_proba_for_pos = clf_list[i].predict_proba(x_test)[:, 1]
            predict_proba_for_neg = clf_list[i].predict_proba(x_test)[:, 0]
            #print(predict_proba_for_pos)
            #print(predict_proba_for_neg)
            #print(predict)


            pd_eval[columns[9]][i] += roc_auc_score(y_test, predict_proba_for_pos,average=None)
            #print(pd_eval[columns[9]][i])
            #fpr_neg, tpr_neg, t2 = roc_curve(y_test, predict_proba_for_neg, pos_label=-1)
            #print(fpr_neg, tpr_neg)
            #print(auc(fpr_neg, tpr_neg))

            auc_mat[k][i] = roc_auc_score(y_test, predict_proba_for_pos,average=None)


            label_mapping = {0: 1,1:0}
            y_test = pd.Series(y_test)
            y_test = np.array(y_test.map(label_mapping))

            #print(y_test)
            pd_eval[columns[10]][i] += roc_auc_score(y_test, predict_proba_for_neg,average=None)
            #print(pd_eval[columns[10]][i])

            label_mapping = {0: 1, 1: 0}
            y_test = pd.Series(y_test)
            y_test = np.array(y_test.map(label_mapping))


        #print(pd_eval)

        columns = ["classifier", 'precision_score_for_pos', 'recall_score_for_pos', 'f1_score_for_pos', \
                   'precision_score_for_neg', 'recall_score_for_neg', 'f1_score_for_neg', \
                   'accuracy_for_pos', 'accuacy_for_neg', \
                   'auc_for_pos', 'auc_for_neg', \
                   'weighted_aver_precision', 'weighted_aver_recall', 'weighted_aver_f1_score', \
                   'weighted_aver_accuracy', 'weighted_aver_auc']

        #all other score is added, then cal the weighted average score
        pd_eval['weighted_aver_precision']=0.5*pd_eval['precision_score_for_pos']+ \
                                           0.5*pd_eval['precision_score_for_neg']

        pd_eval['weighted_aver_recall'] = 0.5 * pd_eval['recall_score_for_pos'] + \
                                          0.5 * pd_eval['recall_score_for_neg']

        pd_eval['weighted_aver_f1_score'] = 0.5 * pd_eval['f1_score_for_pos'] + \
                                            0.5 * pd_eval['f1_score_for_neg']

        pd_eval['weighted_aver_accuracy'] =0.5 * pd_eval['accuracy_for_pos'] + \
                                           0.5 * pd_eval['accuacy_for_neg']

        pd_eval['weighted_aver_auc'] = 0.5 * pd_eval['auc_for_pos'] + \
                                       0.5 * pd_eval['auc_for_neg']

        k+=1

    pd_eval[columns[1:]]= pd_eval[columns[1:]].apply(average)

    model =list(pd_eval['weighted_aver_accuracy'])
    error = []
    for i in range(len(clf_namelist)):
        error.append(pd_eval['weighted_aver_accuracy'][i])

    model1 = list(pd_eval['weighted_aver_f1_score'])
    error1 = []
    for i in range(len(clf_namelist)):
        error1.append(pd_eval['weighted_aver_f1_score'][i])

    model2 = list(pd_eval['weighted_aver_auc'])
    error2 = []
    for i in range(len(clf_namelist)):
        error2.append(pd_eval['weighted_aver_auc'][i])


    #指定计算error bar的mat:
    acc_mat = acc_mat
    acc_mat = acc_mat.T
    e = []
    for j in range(len(clf_namelist)):
        e.append((max(list(acc_mat[j][:])) - min(list(acc_mat[j][:]))))


    acc_mat = f1_mat
    acc_mat = acc_mat.T
    e1 = []
    for j in range(len(clf_namelist)):
        e1.append((max(list(acc_mat[j][:])) - min(list(acc_mat[j][:]))))

    acc_mat = auc_mat
    acc_mat = acc_mat.T
    e2 = []
    for j in range(len(clf_namelist)):
        e2.append((max(list(acc_mat[j][:])) - min(list(acc_mat[j][:]))))

    return model,model1,model2,error,e,error1,e1,error2,e2



models = [[],[],[],[]]
models1 = [[],[],[],[]]
models2 = [[],[],[],[]]
error = [[],[],[],[]]
e = [[],[],[],[]]
error1 = [[],[],[],[]]
e1 = [[],[],[],[]]
error2 = [[],[],[],[]]
e2 = [[],[],[],[]]
labels = ['Naive Beyasian', 'Decision Tree', 'Random Forest', 'KNN', 'Ada Boost', 'Gradient Boosting']
bar_width = 0.2

i=0
for eachfeature_dataset in X:
    print("now using dataset:")
    print(eachfeature_dataset.columns)
    cols=list(eachfeature_dataset.columns)
    #eachfeature_dataset=sklearn.preprocessing.MinMaxScaler().fit_transform(eachfeature_dataset)
    eachfeature_dataset = np.array(eachfeature_dataset)
    models[i],models1[i],models2[i],error[i],e[i],error1[i],e1[i],error2[i],e2[i]=clf(eachfeature_dataset,y,i,cols)
    i = i + 1

print("this is the graph data")

print(models)
print(models1)
print(models2)
print(error)
print(error1)
print(error2)
print(e)
print(e1)
print(e2)

def pic(X,error_li,e_li,title,models):
    color_list = ['steelblue', '#daa520', '#6dc066', 'indianred']
    for j in range(len(X)):
        error=error_li[j]
        e=e_li[j]
        plt.errorbar(np.arange(6)+bar_width*j, error, yerr=e, fmt='--.',ecolor='#4d4a4a',color=color_list[j])
        #改进方法 把图的信息都打印出来 再分别画三张图
    # 绘图
    plt.bar(np.arange(6), models[0], label='Feature Set-1', color='steelblue', alpha=0.8, width=bar_width)  # 一行画一个model的所有clf
    plt.bar(np.arange(6) + bar_width * 1, models[1], label='Feature Set-2', color='#daa520', alpha=0.8,width=bar_width)
    plt.bar(np.arange(6) + bar_width * 2, models[2], label='Feature Set-3', color='#6dc066', alpha=0.8,width=bar_width)
    plt.bar(np.arange(6) + bar_width * 3, models[3], label='Feature Set-4', color='indianred', alpha=0.8,width=bar_width)

    # 添加轴标签
    plt.xlabel('Classifiers')
    #plt.ylabel('Weighted Average Accuracy')
    #plt.ylabel('Weighted Average AUC')
    #plt.ylabel('Weighted Average F1-Measure')
    plt.ylabel(title)
    # 添加标题
    # plt.title('亿万财富家庭数Top5城市分布')
    # 添加刻度标签
    plt.xticks(np.arange(6) + bar_width*1.5, labels)
    # 设置Y轴的刻度范围
    plt.ylim([0.4, 1.0])
    # 为每个条形图添加数值标签
    # 显示图例
    plt.legend()
    # 显示图形
    plt.show()

pic(X,error1,e1,'some title',models1)
#note 多个数据集可以开启多个线程 但是我有点忘了 多线程的框架了 不管了 就单线程跑把 我去玩了

#致命错误 在你调试代码的时候 不要放很多数据 这样你能改动的太少了 效率太低了

# this is the graph data
# [[0.608111396114435, 0.8098919495053704, 0.8371579557704572, 0.813178682274807, 0.8055330508235222, 0.7942472500122648], [0.6459958879703985, 0.8057217723180109, 0.8407411958854236, 0.781058751882462, 0.8008316328751073, 0.7938282722578115], [0.6082435874245113, 0.8104622449363678, 0.8436732861022935, 0.8001252515135289, 0.8054238893720914, 0.7943090424698631], [0.7903415090188901, 0.8432380000098881, 0.864826259250265, 0.829153251957592, 0.8399986167824185, 0.8389438594322709]]
# [[0.5673358017029833, 0.8096060674357061, 0.8370830652490516, 0.8131679195815209, 0.8051396355620856, 0.7938396939568534], [0.6118217853989666, 0.8055433997692554, 0.8406473124120154, 0.7810170947388922, 0.800496780585048, 0.7933855670778476], [0.5678999325811122, 0.8100483942015699, 0.8436218474870836, 0.8000962633674057, 0.8051022050841958, 0.7938827328585851], [0.7892057884365782, 0.8431069037420663, 0.8647620214686775, 0.8291296687520913, 0.8398706831451183, 0.8388422736288387]]
# [[0.7476944338246758, 0.8730585779061354, 0.9004173324058291, 0.863513880664307, 0.8727365178091222, 0.8642612898879545], [0.7683616856805408, 0.8693018134582788, 0.9092475824838491, 0.8433972995015295, 0.8683633553509557, 0.8628881115574103], [0.7446658772239125, 0.8735461337826113, 0.9104047229717821, 0.8584863865477306, 0.8735525603298513, 0.8648110439839394], [0.8776527315246538, 0.9112187609802547, 0.933526040865692, 0.8854592512670638, 0.9059226712560541, 0.9073254067437638]]
# [[0.608111396114435, 0.8098919495053704, 0.8371579557704572, 0.813178682274807, 0.8055330508235222, 0.7942472500122648], [0.6459958879703985, 0.8057217723180109, 0.8407411958854236, 0.781058751882462, 0.8008316328751073, 0.7938282722578115], [0.6082435874245113, 0.8104622449363678, 0.8436732861022935, 0.8001252515135289, 0.8054238893720914, 0.7943090424698631], [0.7903415090188901, 0.8432380000098881, 0.864826259250265, 0.829153251957592, 0.8399986167824185, 0.8389438594322709]]
# [[0.5673358017029833, 0.8096060674357061, 0.8370830652490516, 0.8131679195815209, 0.8051396355620856, 0.7938396939568534], [0.6118217853989666, 0.8055433997692554, 0.8406473124120154, 0.7810170947388922, 0.800496780585048, 0.7933855670778476], [0.5678999325811122, 0.8100483942015699, 0.8436218474870836, 0.8000962633674057, 0.8051022050841958, 0.7938827328585851], [0.7892057884365782, 0.8431069037420663, 0.8647620214686775, 0.8291296687520913, 0.8398706831451183, 0.8388422736288387]]
# [[0.7476944338246758, 0.8730585779061354, 0.9004173324058291, 0.863513880664307, 0.8727365178091222, 0.8642612898879545], [0.7683616856805408, 0.8693018134582788, 0.9092475824838491, 0.8433972995015295, 0.8683633553509557, 0.8628881115574103], [0.7446658772239125, 0.8735461337826113, 0.9104047229717821, 0.8584863865477306, 0.8735525603298513, 0.8648110439839394], [0.8776527315246538, 0.9112187609802547, 0.933526040865692, 0.8854592512670638, 0.9059226712560541, 0.9073254067437638]]
# [[0.0035638970898355105, 0.001950653732602281, 0.0014129059468579452, 0.012094053142134187, 0.004597216364403134, 0.0032581189371573283], [0.012589624630957363, 0.0022458878110501734, 0.0017292281737663062, 0.002636018557570563, 0.0012863770560944676, 0.0007169970476592624], [0.003110501897933382, 0.0023513285533530714, 0.0019822859552930394, 0.002962884858709325, 0.0018135807676086246, 0.0016975959510754368], [0.00201391817798402, 0.0028152678194853786, 0.002277520033741043, 0.002161535217207966, 0.0022986081822016224, 0.0022986081822016224]]
# [[0.002601553689279834, 0.0019868334491005557, 0.001476017801038787, 0.012078471065020357, 0.004689813609855653, 0.0034121304804237695], [0.013885971559075894, 0.0021365228900065425, 0.00171725416776225, 0.002654244312566223, 0.0014308935531019973, 0.0007534166602938264], [0.0034437494459184093, 0.0021224430295672114, 0.00198574279403696, 0.0029757935477603414, 0.0016499359041703965, 0.0017746554199933318], [0.0020629849261775135, 0.0029546942022636857, 0.0022965978340027116, 0.002170228883344949, 0.0023433708741149584, 0.0023682684650667696]]
# [[0.00739659260593073, 0.002412671729224991, 0.002221953485431527, 0.003663256379941582, 0.0022985687143367572, 0.002392839432246996], [0.0030919288857740312, 0.0006585948096274885, 0.0021429948021688405, 0.0024012745101082977, 0.001606269029430285, 0.0021743800992634066], [0.0036677903428012915, 0.0027536712177363976, 0.002233014126623356, 0.0031881006791801836, 0.0016430427130280645, 0.0022022427369134956], [0.00380817456641791, 0.0016320597356763722, 0.0019922773049789733, 0.0012163150385827715, 0.0013907027041323294, 0.0018210076809632714]]

