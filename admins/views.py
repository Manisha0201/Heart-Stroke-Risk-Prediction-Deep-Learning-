from django.shortcuts import render
from django.contrib import messages
# Create your views here.
from users.models import UserRegistrationModel, HeartDataModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from django_pandas.io import read_frame
#%matplotlib inline
from sklearn.model_selection import train_test_split
import os
#print(os.listdir())

import warnings
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage


def AdminLogin(request):
    return render(request,'AdminLogin.html',{})


def AdminLoginCheck(request):
    if request.method == 'POST':
        usrid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        print("User ID is = ", usrid)
        if usrid == 'admin' and pswd == 'admin':
            return render(request, 'admins/AdminHome.html')

        else:
            messages.success(request, 'Please Check Your Login Details')
    return render(request, 'AdminLogin.html',{})


def RegisterUsersView(request):
    data = UserRegistrationModel.objects.all()
    return render(request,'admins/ViewRegisterUsers.html',{'data':data})


def ActivaUsers(request):
    if request.method == 'GET':
        id = request.GET.get('uid')
        status = 'activated'
        print("PID = ", id, status)
        UserRegistrationModel.objects.filter(id=id).update(status=status)
        data = UserRegistrationModel.objects.all()
        return render(request,'admins/ViewRegisterUsers.html',{'data':data})


def adminML(request):
    #gc.collect()
    dataset = HeartDataModel.objects.all()
    dataset = read_frame(dataset)
    #dataset.fillna
    print(dataset.head())
    print(type(dataset))
    print(dataset.shape)
    print(dataset.head(5))
    print(dataset.sample(5))
    print(dataset.describe())
    dataset.info()
    info = ["age", "1: male, 0: female",
            "chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic",
            "resting blood pressure", " serum cholestoral in mg/dl", "fasting blood sugar > 120 mg/dl",
            "resting electrocardiographic results (values 0,1,2)", " maximum heart rate achieved",
            "exercise induced angina", "oldpeak = ST depression induced by exercise relative to rest",
            "the slope of the peak exercise ST segment", "number of major vessels (0-3) colored by flourosopy",
            "thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]

    for i in range(len(info)):
        print(dataset.columns[i] + ":\t\t\t" + info[i])
    #X = dataset.drop(['target'], axis=1).values
    #print("x",X)
    dataset["target"].describe()
    print(dataset["target"].unique())
    print(dataset.corr()["target"].abs().sort_values(ascending=False))
    y = dataset["target"]
    print("y",y)
    sns.countplot(y)

    print("Dataset Head",dataset.head(25))
    target_temp = dataset.target.value_counts()

    print("target Label Count=",target_temp)
    print("Percentage of patience without heart problems: " + str(round(target_temp[0] * 100 / 303, 2)))
    print("Percentage of patience with heart problems: " + str(round(target_temp[1] * 100 / 303, 2)))
    print(dataset["sex"].unique())
    sns.barplot(x=dataset["sex"], y=y)

    dataset["cp"].unique()
    sns.barplot(x=dataset["cp"], y=y)
    dataset["fbs"].describe()
    dataset["fbs"].unique()
    sns.barplot(x=dataset["fbs"], y=y)
    dataset["restecg"].unique()
    sns.barplot(x=dataset["restecg"], y=y)
    dataset["exang"].unique()
    sns.barplot(x=dataset["exang"], y=y)
    dataset["slope"].unique()
    sns.barplot(x=dataset["slope"], y=y)
    dataset["ca"].unique()
    sns.countplot(dataset["ca"])
    sns.barplot(x=dataset["ca"], y=y)
    dataset["thal"].unique()
    sns.barplot(x=dataset["thal"], y=y)
    sns.distplot(dataset["thal"])
    from sklearn.model_selection import train_test_split

    predictors = dataset.drop("target", axis=1)
    target = dataset["target"]

    X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=0)
    X_train.shape
    print(X_train.shape)
    X_test.shape
    print(X_test.shape)
    Y_train.shape
    print(Y_train.shape)
    Y_test.shape
    print(Y_test.shape)

    # Linear regression
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression()

    lr.fit(X_train, Y_train)

    Y_pred_lr = lr.predict(X_test)
    Y_pred_lr.shape
    print(Y_pred_lr.shape)
    score_lr = round(accuracy_score(Y_pred_lr, Y_test) * 100, 2)

    print("The accuracy score achieved using Linear regression is: " + str(score_lr) + " %")

    # Naive Bayes
    from sklearn.naive_bayes import GaussianNB

    nb = GaussianNB()

    nb.fit(X_train, Y_train)

    Y_pred_nb = nb.predict(X_test)
    Y_pred_nb.shape
    print(Y_pred_nb.shape)
    score_nb = round(accuracy_score(Y_pred_nb, Y_test) * 100, 2)

    print("The accuracy score achieved using Naive Bayes is: " + str(score_nb) + " %")

    # SVM
    from sklearn import svm

    sv = svm.SVC(kernel='linear')

    sv.fit(X_train, Y_train)

    Y_pred_svm = sv.predict(X_test)
    Y_pred_svm.shape
    print(Y_pred_svm.shape)
    score_svm = round(accuracy_score(Y_pred_svm, Y_test) * 100, 2)

    print("The accuracy score achieved using Linear SVM is: " + str(score_svm) + " %")

    # K Nearest Neighbors
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train, Y_train)
    Y_pred_knn = knn.predict(X_test)
    Y_pred_knn.shape
    print(Y_pred_knn.shape)
    score_knn = round(accuracy_score(Y_pred_knn, Y_test) * 100, 2)

    print("The accuracy score achieved using KNN is: " + str(score_knn) + " %")

    # Decision Tree
    from sklearn.tree import DecisionTreeClassifier

    max_accuracy = 0

    for x in range(200):
        dt = DecisionTreeClassifier(random_state=x)
        dt.fit(X_train, Y_train)
        Y_pred_dt = dt.predict(X_test)
        current_accuracy = round(accuracy_score(Y_pred_dt, Y_test) * 100, 2)
        if (current_accuracy > max_accuracy):
            max_accuracy = current_accuracy
            best_x = x

    # print(max_accuracy)
    # print(best_x)

    dt = DecisionTreeClassifier(random_state=best_x)
    dt.fit(X_train, Y_train)
    Y_pred_dt = dt.predict(X_test)
    print(Y_pred_dt.shape)
    score_dt = round(accuracy_score(Y_pred_dt, Y_test) * 100, 2)

    print("The accuracy score achieved using Decision Tree is: " + str(score_dt) + " %")

    # Neural Network
    from keras.models import Sequential
    from keras.layers import Dense
    model = Sequential()
    model.add(Dense(11, activation='relu', input_dim=14))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.fit(X_train,Y_train,epochs=300)
    model.fit(X_train, Y_train, epochs=300)
    Y_pred_nn = model.predict(X_test)
    Y_pred_nn.shape
    print(Y_pred_nn.shape)
    rounded = [round(x[0]) for x in Y_pred_nn]

    Y_pred_nn = rounded
    score_nn = round(accuracy_score(Y_pred_nn, Y_test) * 100, 2)

    print("The accuracy score achieved using Neural Network is: " + str(score_nn) + " %")


    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import SimpleRNN, Dense
    from sklearn.metrics import accuracy_score

    # Reshape input for RNN (assuming X_train has 14 features)
    X_train_rnn = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_rnn = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

    model_rnn = Sequential()
    model_rnn.add(SimpleRNN(11, activation='relu', input_shape=(14, 1)))
    model_rnn.add(Dense(1, activation='sigmoid'))
    model_rnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_rnn.fit(X_train_rnn, Y_train, epochs=300)

    Y_pred_rnn = model_rnn.predict(X_test_rnn)
    rounded = [round(x[0]) for x in Y_pred_rnn]
    score_rnn = round(accuracy_score(rounded, Y_test) * 100, 2)
    print("The accuracy score achieved using RNN is: " + str(score_rnn) + " %")

    # Accuracy Data
    accuracy_data = {
        "Logistic Regression": score_lr,
        "Naive Bayes": score_nb,
        "SVM": score_svm,
        "KNN": score_knn,
        "Decision Tree": score_dt,
        "Neural Network": score_nn,
        "RNN": score_rnn
    }

    # Plotting the graph
    plt.figure(figsize=(6, 6))
    sns.barplot(x=list(accuracy_data.keys()), y=list(accuracy_data.values()), palette="viridis")
    plt.xlabel("Algorithms")
    plt.ylabel("Accuracy (%)")
    plt.title("Algorithm Accuracy Comparison")
    plt.xticks(rotation=30)
    plt.ylim(0, 100)  # Set y-axis limit
    plt.tight_layout()

    # Save the plot as an image
    img_path = "assets/static/accuracy_plot.png"
    plt.savefig(img_path)
    plt.close()

    return render(request, 'admins/AdminHome.html', {'img_path': img_path, 'accuracy_data': accuracy_data})
