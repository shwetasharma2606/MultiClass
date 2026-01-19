import streamlit as st
import pandas as pd


st.title("CSV Uploader and Viewer App")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(uploaded_file)
    #START
    last_column_name = train_df.columns[-1]
    print('last column name:  ',last_column_name)
    X = train_df.drop(columns=[last_column_name])
    y = train_df["price_range"]
    
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, stratify=y, test_size= 0.3)
    
    # Display the DataFrame
    #st.write("Data Preview:", df)
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
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(
            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=42
        )
    }

    #Evaluation functions
    def evaluate_model(name, model, X_train, X_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
        # Probabilities for AUC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
        else:
            auc = np.nan
    
        return {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted"),
            "Recall": recall_score(y_test, y_pred, average="weighted"),
            "F1 Score": f1_score(y_test, y_pred, average="weighted"),
            "AUC": auc,
            "MCC": matthews_corrcoef(y_test, y_pred)
        }

    #Confusion matrix
    def plot_confusion_matrix(y_true, y_pred, model_name):
        cm = confusion_matrix(y_true, y_pred)
    
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()

    #END
else:
    st.info("Please upload a CSV file to get started.")

st.title("Select Model")

# Define the options for the dropdown
options = ["Logistic Regression", "Decision Tree Classifier", "K-Nearest Neighbor Classifier","Naive Bayes Classifier - Gaussian", "Ensemble Model - Random Forest","Ensemble Model - XGBoost"]

# Create the dropdown menu using st.selectbox
selected_option = st.selectbox(
    "How would you like to be contacted?", # Label for the dropdown
    options                               # List, tuple, or array of options
)

# Display the selected option
st.write("Selected model :", selected_option)

st.title("Evaluation Metrics")

st.title("Confusion Metrics")
