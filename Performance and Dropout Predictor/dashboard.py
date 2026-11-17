import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Student Dropout Predictor",
    layout="wide",
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stMetric {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    h1 {color: #1e3a8a; font-weight: 700;}
    h2 {color: #2563eb;}
    .prediction-box {
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        margin: 20px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .stButton>button {
        width: 100%;
        background-color: #2563eb;
        color: white;
        font-weight: bold;
        padding: 15px;
        font-size: 18px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# LOAD DATA AND MODEL

@st.cache_resource
def load_model():
    # Placeholder for actual model loading logic
    try:
        with open('student_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model file 'student_model.pkl' not found.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

@st.cache_data
def load_data():
    # Placeholder for actual data loading logic
    try:
        df = pd.read_csv('student_data.csv')
        feature_imp = pd.read_csv('feature_importance.csv')
        return df, feature_imp
    except FileNotFoundError:
        st.error("Data files not found. Ensure 'student_data.csv' and 'feature_importance.csv' are in the directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# Load data and model only once
try:
    model = load_model()
    df, feature_importance = load_data()
except Exception as e:
    st.error(f"An unexpected error occurred during file loading: {e}")
    st.stop()

# ============================================
# SIDEBAR NAVIGATION - ICON CHANGED 
# ============================================
st.sidebar.title("🧭 NAVIGATION") # Changed 🎓 to 🧭
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Select Page:",
    ["Dashboard Overview", "Predict Dropout Risk", "Model Insights"],
    label_visibility="collapsed"
)

# Map page names to titles with updated icons
page_titles = {
    "Dashboard Overview": "📊 Student Performance & Dropout Analysis",
    "Predict Dropout Risk": "⚠️ Student Dropout Risk Prediction Tool", # Changed 🔮 to ⚠️
    "Model Insights": "🧠 Model Performance & Insights", # Changed 📈 to 🧠
}

# ============================================
# PAGE 1: DASHBOARD OVERVIEW
# ============================================
if page == "Dashboard Overview":
    st.title(page_titles[page])
    st.markdown("### SIWES Defense Project - Predictive Analytics Dashboard")
    st.markdown("---")
    
    # Key Metrics Row (Keeping existing delta icons for metrics)
    col1, col2, col3, col4 = st.columns(4)
    
    total_students = len(df)
    graduates = len(df[df['target'] == 'Graduate'])
    dropouts = len(df[df['target'] == 'Dropout'])
    enrolled = len(df[df['target'] == 'Enrolled'])
    
    with col1:
        st.metric("Total Students", f"{total_students:,}")
    with col2:
        st.metric("Graduates", f"{graduates:,}", 
                  delta=f"{graduates/total_students*100:.1f}%", delta_color="normal")
    with col3:
        st.metric("Dropouts", f"{dropouts:,}", 
                  delta=f"{dropouts/total_students*100:.1f}%", delta_color="inverse")
    with col4:
        st.metric("Enrolled", f"{enrolled:,}", 
                  delta=f"{enrolled/total_students*100:.1f}%", delta_color="off")
    
    st.markdown("---")
    
    # First Row of Charts 
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Student Status Distribution")
        status_counts = df['target'].value_counts()
        fig = px.pie(
            values=status_counts.values, 
            names=status_counts.index,
            color_discrete_sequence=['#10b981', '#ef4444', '#f59e0b'],
            hole=0.4
        )
        fig.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            textfont_size=14
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Gender Distribution by Status")
        gender_status = df.groupby(['Gender', 'target']).size().reset_index(name='count')
        gender_status['Gender'] = gender_status['Gender'].map({0: 'Female', 1: 'Male'})
        fig = px.bar(
            gender_status, 
            x='Gender', 
            y='count', 
            color='target',
            barmode='group',
            color_discrete_sequence=['#10b981', '#ef4444', '#f59e0b'],
            text='count'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Second Row of Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1st Semester Grade Distribution")
        fig = px.box(
            df, 
            x='target', 
            y='Curricular units 1st sem (grade)',
            color='target',
            color_discrete_sequence=['#10b981', '#ef4444', '#f59e0b']
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Age Distribution at Enrollment")
        fig = px.histogram(
            df, 
            x='Age at enrollment', 
            color='target',
            nbins=30,
            color_discrete_sequence=['#10b981', '#ef4444', '#f59e0b']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Third Row
    st.subheader("Admission Grade vs Outcome")
    fig = px.scatter(
        df.sample(1000) if len(df) > 1000 else df,
        x='Admission grade',
        y='Previous qualification (grade)',
        color='target',
        size='Age at enrollment',
        color_discrete_sequence=['#10b981', '#ef4444', '#f59e0b'],
        opacity=0.6
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# PAGE 2: PREDICTION
# ============================================
elif page == "Predict Dropout Risk":
    st.title(page_titles[page])
    st.markdown("### Fill in student details to assess dropout risk")
    st.markdown("---")
    
    with st.form("prediction_form"):
        # Personal Information
        st.subheader("Personal Information")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            marital = st.selectbox("Marital Status", 
                options=[1, 2, 3, 4, 5, 6],
                format_func=lambda x: {1: "Single", 2: "Married", 3: "Widower", 
                                       4: "Divorced", 5: "Facto Union", 6: "Legally Separated"}[x],
                index=0)
        
        with col2:
            gender = st.selectbox("Gender", 
                options=[1, 0],
                format_func=lambda x: "Male" if x == 1 else "Female")
        
        with col3:
            age = st.slider("Age at Enrollment", 17, 70, 20)
        
        with col4:
            displaced = st.selectbox("Displaced Student?",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes")
        
        st.markdown("---")
        
        # Academic Background
        st.subheader("Academic Background")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            prev_grade = st.slider("Previous Qualification Grade", 0.0, 200.0, 120.0, 5.0)
        
        with col2:
            admission_grade = st.slider("Admission Grade", 0.0, 200.0, 120.0, 5.0)
        
        with col3:
            attendance = st.selectbox("Class Attendance",
                options=[1, 0],
                format_func=lambda x: "Daytime" if x == 1 else "Evening")
        
        with col4:
            scholarship = st.selectbox("Scholarship Holder?",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes")
        
        st.markdown("---")
        
        # First Semester Performance
        st.subheader("First Semester Performance")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            units_1st_enrolled = st.number_input("Units Enrolled", 0, 30, 6, key="1st_enr")
        
        with col2:
            units_1st_approved = st.number_input("Units Approved", 0, 30, 5, key="1st_app")
        
        with col3:
            units_1st_grade = st.slider("Average Grade", 0.0, 20.0, 12.0, 0.5, key="1st_grade")
        
        with col4:
            units_1st_eval = st.number_input("Units Evaluated", 0, 40, 6, key="1st_eval")
        
        st.markdown("---")
        
        # Second Semester Performance
        st.subheader("Second Semester Performance")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            units_2nd_enrolled = st.number_input("Units Enrolled", 0, 30, 6, key="2nd_enr")
        
        with col2:
            units_2nd_approved = st.number_input("Units Approved", 0, 30, 5, key="2nd_app")
        
        with col3:
            units_2nd_grade = st.slider("Average Grade", 0.0, 20.0, 12.0, 0.5, key="2nd_grade")
        
        with col4:
            units_2nd_eval = st.number_input("Units Evaluated", 0, 40, 6, key="2nd_eval")
        
        st.markdown("---")
        
        # Financial Status
        st.subheader("Financial Status")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            debtor = st.selectbox("Outstanding Debts?",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes")
        
        with col2:
            tuition_updated = st.selectbox("Tuition Up to Date?",
                options=[1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No")
        
        with col3:
            special_needs = st.selectbox("Special Educational Needs?",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes")
        
        st.markdown("---")
        
        # Submit Button
        submitted = st.form_submit_button("PREDICT DROPOUT RISK")
        
        if submitted:
            # Prepare input (all 36 features in correct order)
            input_array = np.array([[
                marital, 1, 1, 1, attendance, 1, prev_grade,
                1, 1, 1, 1, 1,
                admission_grade, displaced, special_needs, debtor, tuition_updated,
                gender, scholarship, age, 0,
                0, units_1st_enrolled, units_1st_eval, units_1st_approved, units_1st_grade, 0,
                0, units_2nd_enrolled, units_2nd_eval, units_2nd_approved, units_2nd_grade, 0,
                10.8, 1.4, 1.74
            ]])
            
            # Make prediction
            prediction = model.predict(input_array)[0]
            probabilities = model.predict_proba(input_array)[0]
            
            # Display Results
            st.markdown("---")
            st.markdown("## Prediction Results")
            
            # Prediction Box
            if prediction == "Dropout":
                st.markdown("""
                    <div class="prediction-box" style="background-color: #fee2e2; color: #991b1b; border: 3px solid #dc2626;">
                        HIGH RISK: LIKELY TO DROPOUT
                    </div>
                """, unsafe_allow_html=True)
                st.warning("**Recommendation**: Immediate intervention required. Student needs academic counseling and support services.")
            
            elif prediction == "Graduate":
                st.markdown("""
                    <div class="prediction-box" style="background-color: #d1fae5; color: #065f46; border: 3px solid #10b981;">
                        LOW RISK: LIKELY TO GRADUATE
                    </div>
                """, unsafe_allow_html=True)
                st.success("**Recommendation**: Student is on track. Continue monitoring and provide regular encouragement.")
            
            else:
                st.markdown("""
                    <div class="prediction-box" style="background-color: #fef3c7; color: #92400e; border: 3px solid #f59e0b;">
                        MODERATE RISK: CURRENTLY ENROLLED
                    </div>
                """, unsafe_allow_html=True)
                st.info("**Recommendation**: Monitor closely. Provide additional support if performance declines.")
            
            # Probability Chart
            col1, col2 = st.columns([2, 1])
            
            with col1:
                prob_df = pd.DataFrame({
                    'Status': model.classes_,
                    'Probability': probabilities * 100
                })
                
                fig = px.bar(
                    prob_df, 
                    x='Status', 
                    y='Probability',
                    color='Status',
                    text='Probability',
                    color_discrete_sequence=['#3b82f6', '#ef4444', '#10b981']
                )
                fig.update_traces(
                    texttemplate='%{text:.1f}%', 
                    textposition='outside',
                    textfont_size=16
                )
                fig.update_layout(
                    showlegend=False,
                    yaxis_title="Probability (%)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### Key Metrics")
                max_prob = max(probabilities) * 100
                st.metric("Confidence Level", f"{max_prob:.1f}%")
                
                risk_level = "High" if prediction == "Dropout" else "Low" if prediction == "Graduate" else "Moderate"
                st.metric("Risk Category", risk_level)
                
                st.metric("Model", "Random Forest")

# ============================================
# PAGE 3: MODEL INSIGHTS
# ============================================
else:
    st.title(page_titles[page])
    st.markdown("### Understanding the Prediction Model")
    st.markdown("---")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Top 15 Most Important Features")
        fig = px.bar(
            feature_importance.head(15),
            x='importance',
            y='feature',
            orientation='h',
            color='importance',
            color_continuous_scale='Blues',
            text='importance'
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Model Information")
        
        st.markdown("""
        **Model Type:**
        Random Forest Classifier
        
        **Total Features:**
        36 student attributes
        
        **Training Data:**
        """ + f"{int(len(df) * 0.8):,} students" + """
        
        **Test Accuracy:**
        View Jupyter notebook results
        """)
        
        st.markdown("---")
        
        st.subheader("Key Factors")
        st.markdown("""
        **Academic Performance:**
        - Semester grades & approved units
        - Previous qualification scores
        - Curricular evaluations
        
        **Financial Indicators:**
        - Debt status
        - Tuition payment status
        - Scholarship availability
        
        **Demographics:**
        - Age at enrollment
        - Gender & marital status
        - Displaced status
        
        **Economic Context:**
        - Unemployment rate
        - GDP & inflation
        """)
    
    st.markdown("---")
    
    # Feature Importance Table
    st.subheader("Complete Feature Importance Table")
    st.dataframe(
        feature_importance.style.background_gradient(subset=['importance'], cmap='Blues'),
        height=400,
        use_container_width=True
    )


st.sidebar.markdown("---")
st.sidebar.info("💬 **Need Help?**\n\nUse the navigation menu above to explore different features.")