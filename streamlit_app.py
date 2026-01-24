import streamlit as st

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef
)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
train_df = None
disp = None


def handle_change_model():
    if train_df is not None:
        results = []
        #st.empty()
        trainModel(train_df)

st.title("Multiclass CLassification implementation with different models")

st.title("Select Model")

# Define the options for the dropdown
options = [ "All","Logistic Regression", "Decision Tree", "K-Nearest Neighbor Classifier","Naive Bayes Classifier - Gaussian", "Ensemble Model - Random Forest","Ensemble Model - XGBoost"]

# Create the dropdown menu using st.selectbox
selected_option = st.selectbox(
    "Select the model for training.", # Label for the dropdown
    options,                               # List, tuple, or array of options
    on_change=handle_change_model
)

# Display the selected option
st.write("Selected model :", selected_option)

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
results = []
n_classes = 4

#Confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name, class_names):
    cm = confusion_matrix(y_true, y_pred)
    # 2. Create the plot
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    #plt.title('Confusion Matrix ')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    disp.plot(cmap="Blues", ax=ax, colorbar=False) # Plot onto the specified axes

    # 3. Display in a container
    with container:
        st.write("### Confusion Matrix : ",model_name)
        st.pyplot(fig)
     

#Evaluation functions
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Probabilities for AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
    else:
        auc = np.nan

    
container = st.container(border=True, )
#container_CM = st.container(border=True)
def trainModel(train_df):
    #START
    #st.empty()
    #container.empty()
    results = []

    last_column_name = train_df.columns[-1]
    #print('last column name:  ',last_column_name)
    X = train_df.drop(columns=[last_column_name])
    y = train_df["price_range"]
    
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, stratify=y, test_size= 0.3)
       
    #Encode catagorical
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test  = le.transform(y_test)
    
    n_classes = len(np.unique(y_train))

    #Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    #Defining models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbor Classifier": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes Classifier - Gaussian": GaussianNB(),
        "Ensemble Model - Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Ensemble Model - XGBoost": XGBClassifier(
            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=42
        )
        
    }
    for name, model in models.items():
        
        if(name==selected_option or "All"==selected_option):
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            # Metrics
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test_scaled)
                auc_score = roc_auc_score(y_test, y_prob, multi_class="ovr")
            else:
                auc_score = np.nan
            
            results.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average="weighted"),
                "Recall": recall_score(y_test, y_pred, average="weighted"),
                "F1 Score": f1_score(y_test, y_pred, average="weighted"),
                "AUC": auc_score,
                "MCC": matthews_corrcoef(y_test, y_pred)
            })
            class_names = [f'Class {i}' for i in model.classes_] # Get the class names
            st.empty()
            container.empty()
            # Confusion Matrix
            plot_confusion_matrix(y_test, y_pred, name, class_names)

            evaluate_model(name,model,X_train, X_test, y_train, y_test)
        
    
    
    #Result comparision
    results_df = pd.DataFrame(results)
    container.write("EVALUATION METRICS")
    container.dataframe(results_df) 

    #END

if uploaded_file is not None:
    # Read the CSV file into a pandas DataFrame
    train_df = pd.read_csv(uploaded_file)
    trainModel(train_df)
else:
    st.info("Please upload a CSV file to get started.")






