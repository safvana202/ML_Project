import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_excel('loan_data.xlsx')

st.header("📊 Dataset Preview")
st.dataframe(df)
st.divider()
# st.dataframe(df.head())
st.subheader("Features")
# st.dataframe(df.columns)
# st.dataframe(df.isna().sum())
# st.dataframe(df.value_counts())
st.dataframe(df.describe())
# st.dataframe(df.dtypes)
df.drop(["person_education"],axis=1,inplace=True)

# EDA - Exploratory Data Analysis

st.header("📊 Loan Status Distribution")

loan_counts = df['loan_status'].value_counts()
st.bar_chart(loan_counts)
                               
st.write("Loan Status Count:")
st.dataframe(loan_counts)

# df['loan_status'].value_counts().plot(kind='bar')
# plt.title("Loan Status Distribution")
# plt.xlabel("Loan Status")
# plt.ylabel("Count")
# plt.show()

# print("\nMissing Values (%):\n", df.isnull().mean() * 100)

num_cols = ['person_income', 'person_emp_exp', 'loan_amnt']

for col in num_cols:
    st.header(f"📊Distribution of {col}")
    data = df[col].dropna()
    bins = st.slider(f"Select number of bins {col}", 5, 50, 20,key=col )
    fig = px.histogram(df, x=col, nbins=bins)
    st.plotly_chart(fig, use_container_width=True)


    # plt.hist(df[col].dropna(), bins=30)
    # plt.title(f"Distribution of {col}")
    # plt.xlabel(col)
    # plt.ylabel("Frequency")
    # plt.show()

for col in num_cols:
    st.header(f"📦 Boxplot of {col}")
    fig, ax = plt.subplots()
    ax.boxplot(df[col].dropna())
    ax.set_title(f"Boxplot of {col}")
    ax.set_ylabel(col)
    st.pyplot(fig)
    # plt.boxplot(df[col].dropna())
    # plt.title(f"Boxplot of {col}")
    # plt.ylabel(col)
    # plt.show()

st.header("📊 Credit Score vs Loan Status")
cross_tab = pd.crosstab(df['credit_score'], df['loan_status'])
fig, ax = plt.subplots()
cross_tab.plot(kind='bar', ax=ax)
ax.set_title("Credit Score vs Loan Status")
ax.set_xlabel("Credit Score")
ax.set_ylabel("Count")
st.pyplot(fig)   

# plt.crosstab(df['credit_score'], df['loan_status']).plot(kind='bar')
# plt.title("Credit History vs Loan Status")
# plt.xlabel("Credit History")
# plt.ylabel("Count")
# plt.show()

corr = df[num_cols].corr()
st.write("\nCorrelation Matrix:\n", corr)
# print("\nCorrelation Matrix:\n", corr)

df['person_gender'] = df['person_gender'].replace({
    'male': 'Male',
    'm': 'Male',
    'female': 'Female',
    'f': 'Female'
})

#encoding:
df["person_gender"]=df["person_gender"].map({"Male":0,"Female":1})

# st.dataframe(df['person_home_ownership'].value_counts())

df["person_home_ownership"]=df["person_home_ownership"].map({"RENT":0,"MORTGAGE":1,"OWN":2,"OTHER":3})

# st.dataframe(df['loan_intent'].value_counts())

df["loan_intent"]=df["loan_intent"].map({"EDUCATION":0,"MEDICAL":1,"VENTURE":2,"PERSONAL":3,"DEBTCONSOLIDATION":4,"HOMEIMPROVEMENT":5})
# st.dataframe(df)

# st.dataframe(df.dtypes)
# st.dataframe(df['previous_loan_defaults_on_file'].value_counts())

df["previous_loan_defaults_on_file"]=df["previous_loan_defaults_on_file"].map({"Yes":0,"No":1})
# st.dataframe(df.head())

X=df.iloc[:,1:-1]
st.dataframe(X.head())

y=df.iloc[:,-1]
st.dataframe(y.head())

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
X_scaled

X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.3,random_state=1)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name}: {acc:.4f}")

results_df = pd.DataFrame(
    results.items(),
    columns=["Model", "Accuracy"]
).sort_values(by="Accuracy", ascending=False)
results_df


rf=RandomForestClassifier(n_estimators=50)
rf.fit(X_train,y_train)

y_pred=rf.predict(X_test)
y_pred

# print(classification_report(y_test,y_pred))

y_new=rf.predict(scaler.transform([[0,71948,0,0,35000,3,16.02,0.49,3,561,0]]))
y_new

import joblib
# Save model
joblib.dump(rf, "random_forest_model.save")

# Load model
model = joblib.load("random_forest_model.save")

# Predict
model.predict(X_test)

