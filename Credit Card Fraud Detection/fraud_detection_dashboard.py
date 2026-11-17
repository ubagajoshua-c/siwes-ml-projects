import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# Page configuration
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")

# Load models and data
@st.cache_resource
def load_models():
    with open('lr_model.pkl', 'rb') as f:
        lr_model = pickle.load(f)
    with open('rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    return lr_model, rf_model

@st.cache_data
def load_data():
    with open('test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)
    df = pd.read_csv('creditcard.csv')
    return test_data, df

try:
    lr_model, rf_model = load_models()
    test_data, df = load_data()
    y_test = test_data['y_test']
    y_pred_lr = test_data['y_pred_lr']
    y_pred_rf = test_data['y_pred_rf']
    y_pred_proba_lr = test_data['y_pred_proba_lr']
    y_pred_proba_rf = test_data['y_pred_proba_rf']
except FileNotFoundError:
    st.error("⚠️ Please run the Jupyter notebook first to train the models!")
    st.stop()

# Sidebar
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Select Page", ["Overview", "Model Results", "Predictions"])

fraud_rate = (df['Class'].sum()/len(df)*100)
st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset Info")
st.sidebar.write(f"**Total:** {len(df):,} transactions")
st.sidebar.write(f"**Fraud:** {df['Class'].sum():,} cases")
st.sidebar.write(f"**Rate:** {fraud_rate:.2f}%")

# Main Title
st.title("💳 Credit Card Fraud Detection")
st.markdown("**SIWES Project Dashboard**")
st.markdown("---")

# PAGE 1: OVERVIEW
if page == "Overview":
    st.header("📊 Project Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", f"{len(df):,}")
    with col2:
        st.metric("Normal", f"{(df['Class']==0).sum():,}")
    with col3:
        st.metric("Fraud", f"{df['Class'].sum():,}")
    with col4:
        st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Class Distribution")
        fig, ax = plt.subplots(figsize=(6, 5))
        class_counts = df['Class'].value_counts()
        colors = ['#2ecc71', '#e74c3c']
        wedges, texts, autotexts = ax.pie(class_counts, labels=['Normal', 'Fraud'], 
                                            autopct='%1.1f%%', colors=colors, startangle=90)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(12)
            autotext.set_weight('bold')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Transaction Counts")
        fig, ax = plt.subplots(figsize=(6, 5))
        bars = ax.bar(['Normal', 'Fraud'], class_counts.values, color=colors, edgecolor='black')
        ax.set_ylabel('Count')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}', ha='center', va='bottom')
        st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("💰 Transaction Amounts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        stats = pd.DataFrame({
            'Normal': [
                f"${df[df['Class']==0]['Amount'].mean():.2f}",
                f"${df[df['Class']==0]['Amount'].median():.2f}",
                f"${df[df['Class']==0]['Amount'].max():.2f}"
            ],
            'Fraud': [
                f"${df[df['Class']==1]['Amount'].mean():.2f}",
                f"${df[df['Class']==1]['Amount'].median():.2f}",
                f"${df[df['Class']==1]['Amount'].max():.2f}"
            ]
        }, index=['Average', 'Median', 'Maximum'])
        st.dataframe(stats, use_container_width=True)
    
    with col2:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.hist(df[df['Class']==0]['Amount'], bins=50, alpha=0.6, label='Normal', color='#2ecc71')
        ax.hist(df[df['Class']==1]['Amount'], bins=50, alpha=0.6, label='Fraud', color='#e74c3c')
        ax.set_xlabel('Amount ($)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.set_xlim(0, 500)
        st.pyplot(fig)

# PAGE 2: MODEL RESULTS
elif page == "Model Results":
    st.header("🤖 Model Performance")
    
    # Calculate metrics
    lr_acc = accuracy_score(y_test, y_pred_lr)
    lr_prec = precision_score(y_test, y_pred_lr)
    lr_rec = recall_score(y_test, y_pred_lr)
    lr_f1 = f1_score(y_test, y_pred_lr)
    lr_auc = roc_auc_score(y_test, y_pred_proba_lr)
    
    rf_acc = accuracy_score(y_test, y_pred_rf)
    rf_prec = precision_score(y_test, y_pred_rf)
    rf_rec = recall_score(y_test, y_pred_rf)
    rf_f1 = f1_score(y_test, y_pred_rf)
    rf_auc = roc_auc_score(y_test, y_pred_proba_rf)
    
    # Display metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Logistic Regression")
        st.metric("Accuracy", f"{lr_acc:.3f}")
        st.metric("Precision", f"{lr_prec:.3f}")
        st.metric("Recall", f"{lr_rec:.3f}")
        st.metric("F1-Score", f"{lr_f1:.3f}")
        st.metric("ROC-AUC", f"{lr_auc:.3f}")
    
    with col2:
        st.subheader("Random Forest")
        st.metric("Accuracy", f"{rf_acc:.3f}")
        st.metric("Precision", f"{rf_prec:.3f}")
        st.metric("Recall", f"{rf_rec:.3f}")
        st.metric("F1-Score", f"{rf_f1:.3f}")
        st.metric("ROC-AUC", f"{rf_auc:.3f}")
    
    st.info("""
    **What these mean:**
    - **Accuracy**: Overall correct predictions
    - **Precision**: When we predict fraud, how often are we right
    - **Recall**: How many actual frauds did we catch
    - **F1-Score**: Balance between Precision and Recall
    - **ROC-AUC**: Overall model quality (higher is better)
    """)
    
    st.markdown("---")
    
    # Comparison Chart
    st.subheader("Metrics Comparison")
    
    metrics_data = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
        'Logistic Regression': [lr_acc, lr_prec, lr_rec, lr_f1, lr_auc],
        'Random Forest': [rf_acc, rf_prec, rf_rec, rf_f1, rf_auc]
    })
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics_data))
    width = 0.35
    
    ax.bar(x - width/2, metrics_data['Logistic Regression'], width, 
           label='Logistic Regression', color='#3498db')
    ax.bar(x + width/2, metrics_data['Random Forest'], width, 
           label='Random Forest', color='#2ecc71')
    
    ax.set_ylabel('Score')
    ax.set_xlabel('Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_data['Metric'])
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Confusion Matrices
    st.subheader("Confusion Matrix")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Logistic Regression**")
        cm_lr = confusion_matrix(y_test, y_pred_lr)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm_lr, cmap='Blues')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Normal', 'Fraud'])
        ax.set_yticklabels(['Normal', 'Fraud'])
        
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm_lr[i, j]:,}', ha='center', va='center', 
                       color='white' if cm_lr[i, j] > cm_lr.max()/2 else 'black', 
                       fontsize=14, weight='bold')
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)
    
    with col2:
        st.markdown("**Random Forest**")
        cm_rf = confusion_matrix(y_test, y_pred_rf)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm_rf, cmap='Greens')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Normal', 'Fraud'])
        ax.set_yticklabels(['Normal', 'Fraud'])
        
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm_rf[i, j]:,}', ha='center', va='center', 
                       color='white' if cm_rf[i, j] > cm_rf.max()/2 else 'black', 
                       fontsize=14, weight='bold')
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)
    
    st.info("**Confusion Matrix shows:** Top-left = Correct Normal predictions, Bottom-right = Correct Fraud predictions")
    
    st.markdown("---")
    
    # ROC Curve
    st.subheader("ROC Curve")
    
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC={lr_auc:.3f})', 
            color='#3498db', linewidth=2)
    ax.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={rf_auc:.3f})', 
            color='#2ecc71', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random Guess', linewidth=1)
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve - Model Performance')
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    
    st.info("**ROC Curve shows:** How well the model separates fraud from normal. Closer to top-left is better.")

# PAGE 3: PREDICTIONS
elif page == "Predictions":
    st.header("🔍 Prediction Analysis")
    
    model_choice = st.selectbox("Select Model", ["Random Forest", "Logistic Regression"])
    
    if model_choice == "Random Forest":
        y_pred = y_pred_rf
        y_pred_proba = y_pred_proba_rf
    else:
        y_pred = y_pred_lr
        y_pred_proba = y_pred_proba_lr
    
    # Summary
    col1, col2, col3 = st.columns(3)
    
    correct = (y_test.values == y_pred).sum()
    with col1:
        st.metric("Total Predictions", len(y_pred))
    with col2:
        st.metric("Correct", correct)
    with col3:
        st.metric("Accuracy", f"{(correct/len(y_pred)*100):.1f}%")
    
    st.markdown("---")
    
    # Confidence Distribution
    st.subheader("Prediction Confidence")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(y_pred_proba, bins=50, color='#9b59b6', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Fraud Probability (0=Normal, 1=Fraud)')
    ax.set_ylabel('Count')
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold')
    ax.legend()
    st.pyplot(fig)
    
    st.info("Values below 0.5 are classified as Normal, above 0.5 as Fraud")
    
    st.markdown("---")
    
    # Sample Predictions
    st.subheader("Sample Predictions")
    
    pred_df = pd.DataFrame({
        'Actual': ['Normal' if x == 0 else 'Fraud' for x in y_test.values],
        'Predicted': ['Normal' if x == 0 else 'Fraud' for x in y_pred],
        'Fraud Probability': [f"{p:.3f}" for p in y_pred_proba],
        'Result': ['✅ Correct' if y_test.values[i] == y_pred[i] else '❌ Wrong' 
                   for i in range(len(y_pred))]
    })
    
    filter_choice = st.radio("Show", ["All", "Correct Only", "Wrong Only"])
    
    if filter_choice == "Correct Only":
        display_df = pred_df[pred_df['Result'] == '✅ Correct']
    elif filter_choice == "Wrong Only":
        display_df = pred_df[pred_df['Result'] == '❌ Wrong']
    else:
        display_df = pred_df
    
    st.dataframe(display_df.head(50), use_container_width=True)
    st.write(f"Showing 50 of {len(display_df)} predictions")
    
    st.markdown("---")
    
    # Error Analysis
    st.subheader("Error Analysis")
    
    false_pos = ((y_test == 0) & (y_pred == 1)).sum()
    false_neg = ((y_test == 1) & (y_pred == 0)).sum()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("False Positives", int(false_pos))
        st.write("Normal flagged as Fraud")
    
    with col2:
        st.metric("False Negatives", int(false_neg))
        st.write("Fraud missed (worst case)")

# Footer
st.markdown("---")
st.markdown("**Credit Card Fraud Detection | SIWES Project**")