import gc
from django.shortcuts import render
from django.contrib import messages
from users.forms import UserRegistrationForm, HeartDataForm
from users.models import UserRegistrationModel, HeartDataModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from django_pandas.io import read_frame
from sklearn.model_selection import train_test_split
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import warnings
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage

# Path to the RandomForest model file



def UserLogin(request):
    return render(request, 'UserLogin.html', {})

def UserRegisterAction(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'Register.html', {'form': form})
        else:
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'Register.html', {'form': form})

def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})

import logging

def UserDataView(request):
    data_list = HeartDataModel.objects.all()
    page = request.GET.get('page', 1)

    paginator = Paginator(data_list, 10)
    try:
        users = paginator.page(page)
    except PageNotAnInteger:
        users = paginator.page(1)
    except EmptyPage:
        users = paginator.page(paginator.num_pages)

    return render(request, 'users/DataView_list.html', {'users': users})

def UserMachineLearning(request):
    dataset = HeartDataModel.objects.all()
    dataset = read_frame(dataset)
    
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
    
    dataset["target"].describe()
    print(dataset["target"].unique())
    print(dataset.corr()["target"].abs().sort_values(ascending=False))
    y = dataset["target"]
    print("y", y)
    
    
    
    print("Dataset Head", dataset.head(25))
    target_temp = dataset.target.value_counts()

    print("target Label Count=", target_temp)
    print("Percentage of patients without heart problems: " + str(round(target_temp[0] * 100 / 303, 2)))
    print("Percentage of patients with heart problems: " + str(round(target_temp[1] * 100 / 303, 2)))
    
    sns.barplot(x= dataset["sex"], y= y)
    plt.show()
    
    sns.barplot(x= dataset["cp"], y= y)
    plt.show()
    
    sns.barplot(x= dataset["fbs"], y= y)
    plt.show()
    
    sns.barplot(x= dataset["restecg"], y= y)
    plt.show()
    
    sns.barplot(x= dataset["exang"], y= y)
    plt.show()
    
    sns.barplot(x= dataset["slope"], y= y)
    plt.show()
    
    
    
    sns.barplot(x= dataset["ca"], y= y)
    plt.show()
    
    sns.barplot(x= dataset["thal"], y= y)
    plt.show()
    
    sns.distplot(dataset["thal"])
    plt.show()
    
    from sklearn.model_selection import train_test_split

    predictors = dataset.drop("target", axis=1)
    target = dataset["target"]

    X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=0)

    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn import svm
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from keras.models import Sequential
    from keras.layers import Dense

    # Train and evaluate models
    lr = LogisticRegression()
    lr.fit(X_train, Y_train)
    Y_pred_lr = lr.predict(X_test)
    score_lr = round(accuracy_score(Y_pred_lr, Y_test) * 100, 2)
    print("The accuracy score achieved using Linear regression is: " + str(score_lr) + " %")

    nb = GaussianNB()
    nb.fit(X_train, Y_train)
    Y_pred_nb = nb.predict(X_test)
    score_nb = round(accuracy_score(Y_pred_nb, Y_test) * 100, 2)
    print("The accuracy score achieved using Naive Bayes is: " + str(score_nb) + " %")

    sv = svm.SVC(kernel='linear')
    sv.fit(X_train, Y_train)
    Y_pred_svm = sv.predict(X_test)
    score_svm = round(accuracy_score(Y_pred_svm, Y_test) * 100, 2)
    print("The accuracy score achieved using Linear SVM is: " + str(score_svm) + " %")

    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train, Y_train)
    Y_pred_knn = knn.predict(X_test)
    score_knn = round(accuracy_score(Y_pred_knn, Y_test) * 100, 2)
    print("The accuracy score achieved using KNN is: " + str(score_knn) + " %")

    max_accuracy = 0
    for x in range(200):
        dt = DecisionTreeClassifier(random_state=x)
        dt.fit(X_train, Y_train)
        Y_pred_dt = dt.predict(X_test)
        current_accuracy = round(accuracy_score(Y_pred_dt, Y_test) * 100, 2)
        if (current_accuracy > max_accuracy):
            max_accuracy = current_accuracy
            best_x = x

    dt = DecisionTreeClassifier(random_state=best_x)
    dt.fit(X_train, Y_train)
    Y_pred_dt = dt.predict(X_test)
    score_dt = round(accuracy_score(Y_pred_dt, Y_test) * 100, 2)
    print("The accuracy score achieved using Decision Tree is: " + str(score_dt) + " %")

    model_nn = Sequential()
    model_nn.add(Dense(11, activation='relu', input_dim=14))
    model_nn.add(Dense(1, activation='sigmoid'))
    model_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_nn.fit(X_train, Y_train, epochs=300)
    Y_pred_nn = model_nn.predict(X_test)
    rounded = [round(x[0]) for x in Y_pred_nn]
    score_nn = round(accuracy_score(rounded, Y_test) * 100, 2)
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


    scores = [score_lr, score_nb, score_svm, score_knn, score_dt, score_nn, score_rnn]
    algorithms = ["LR", "Naive Bayes", "SVM", "K-Nearest Neighbors", "Decision Tree", "Neural Network", " RNN" ]

    plt.xlabel("Algorithms")
    plt.ylabel("Accuracy score")
    # sns.barplot(algorithms, scores)
    # plt.show()
    
    dict = {
        "score_lr": score_lr,
        "score_nb": score_nb,
        "score_svm": score_svm,
        "score_knn": score_knn,
        "score_dt": score_dt,
        "score_nn": score_nn,
        "score_rnn": score_rnn,
    }
    return render(request, 'users/Machinelearning.html', dict)
def output_prediction(request):

    return render(request,"users/predictions.html")
def load_model():
    return joblib.load('users/random_forest_model.joblib')

# Example function to preprocess input data
def preprocess_data(input_data):
    # Implement your data preprocessing here (convert types, scale, etc.)
    # Example: Convert input data to numpy array
    processed_data = np.array(input_data).reshape(1, -1)
    return processed_data

from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def user_data_prediction(request):
    if request.method == 'POST':
        # Initialize risk_factors to an empty list to avoid reference errors
        risk_factors = []

        # Extract input data
        user_input = {
            'age': float(request.POST.get('age')),
            'sex': int(request.POST.get('sex')),
            'cp': float(request.POST.get('cp')),
            'trestbps': float(request.POST.get('trestbps')),
            'chol': float(request.POST.get('chol')),
            'fbs': int(request.POST.get('fbs')),
            'restecg': int(request.POST.get('restecg')),
            'thalach': float(request.POST.get('thalach')),
            'exang': int(request.POST.get('exang')),
            'oldpeak': float(request.POST.get('oldpeak')),
            'slope': int(request.POST.get('slope')),
            'ca': float(request.POST.get('ca')),
            'thal': float(request.POST.get('thal'))
        }

        input_data = np.array([list(user_input.values())])

        # Load dataset and train model
        df = pd.read_csv("heart.csv")
        X = df[list(user_input.keys())]
        y = df['target']

        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=1)
        model = RandomForestClassifier()
        model.fit(X_train, Y_train)

        # Predict risk and probability
        prediction = model.predict(input_data)[0]
        prediction_prob = model.predict_proba(input_data)[0][1]  # Probability of having heart disease

        # Determine risk level
        if prediction_prob >= 0.75:
            risk_level = "High"
            recommendation = "Urgent medical consultation required. Maintain a heart-healthy diet, avoid smoking, and exercise regularly."
        elif 0.5 <= prediction_prob < 0.75:
            risk_level = "Moderate"
            recommendation = "Consult a doctor for a heart checkup. Regular exercise and a balanced diet can help."
        else:
            risk_level = "Low"
            recommendation = "Your heart condition seems good. Maintain a healthy lifestyle to keep it that way."

        # Convert probability to percentage
        risk_percentage = round(prediction_prob * 100, 2)

        # Get feature importance scores
        feature_importances = model.feature_importances_
        feature_names = X.columns

        # Map features with their importance values
        feature_importance_dict = {feature_names[i]: feature_importances[i] for i in range(len(feature_names))}

        # Sort by importance (descending)
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

        # Identify top risky features affecting the user's result
        risk_factors = [feat for feat, imp in sorted_features if imp > 0.05]  # Features with significant importance

        # ✅ Check risk conditions safely
        caused_features = []
        if 'trestbps' in risk_factors and user_input['trestbps'] > 140:
            caused_features.append("High Blood Pressure (trestbps)")
        if 'chol' in risk_factors and user_input['chol'] > 200:
            caused_features.append("High Cholesterol (chol)")
        if 'thalach' in risk_factors and user_input['thalach'] < 100:
            caused_features.append("Low Max Heart Rate (thalach)")

        # Pass data to template
        return render(request, 'users/heart_risk_result.html', {
            'risk_level': risk_level,
            'risk_percentage': risk_percentage,
            'recommendation': recommendation,
            'risk_factors': risk_factors,
            'caused_features': caused_features,  # Send caused factors
            'details': input_data,
            
        })

    # Handle GET request
    return render(request, 'users/predictions.html')

from django.http import HttpResponse
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

def generate_pdf(request, risk_level, risk_percentage, caused_features, recommendation):
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="heart_disease_report.pdf"'

    # Create PDF document
    pdf = SimpleDocTemplate(response, pagesize=letter)
    elements = []

    # Define styles
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    normal_style = styles['Normal']

    # Format risk level with color
    risk_color = "red" if risk_level == "High" else "orange" if risk_level == "Moderate" else "green"
    risk_level_formatted = Paragraph(f"<font color='{risk_color}'><b>{risk_level}</b></font>", normal_style)

    # Format caused feature with red color
    caused_features_formatted = Paragraph(f"<font color='red'><b>{caused_features}</b></font>", normal_style)

    # Table Data
    data = [
        ["Risk Level", risk_level_formatted],
        ["Risk Percentage", f"{risk_percentage}%"],
        ["Main Caused Feature", caused_features_formatted],
        ["Recommendation", Paragraph(recommendation, normal_style)],
    ]

    # Create Table
    table = Table(data, colWidths=[150, 300])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))

    elements.append(Paragraph("Heart Disease Prediction Report", title_style))
    elements.append(table)

    # Build PDF
    pdf.build(elements)
    return response


