import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configuration de la page
st.set_page_config(
    page_title="COVID-19 Analysis Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .prediction-positive {
        background-color: #ffcccc;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #ff4444;
        text-align: center;
    }
    .prediction-negative {
        background-color: #ccffcc;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #44ff44;
        text-align: center;
    }
    .feature-importance {
        background-color: #e6f3ff;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<h1 class="main-header">🦠 COVID-19 Analysis & Prediction Dashboard</h1>', unsafe_allow_html=True)

# --- 1. Charger le dataset ---
st.sidebar.header("📁 Data Configuration")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Nettoyage rapide
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    df = df.replace({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})
    
    # Vérifier si la colonne cible existe
    if 'COVID-19' not in df.columns:
        st.error("❌ Target column 'COVID-19' not found in dataset!")
        st.stop()
else:
    st.warning("📝 Please upload a CSV file to begin analysis.")
    st.stop()

# --- Variables globales pour stocker X et y ---
if 'X' not in st.session_state:
    st.session_state.X = None
if 'y' not in st.session_state:
    st.session_state.y = None

# --- Layout avec colonnes pour les métriques ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_patients = len(df)
    st.markdown(f"""
    <div class="metric-card">
        <h3>Total Patients</h3>
        <h2>{total_patients}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    covid_positive = df['COVID-19'].sum()
    st.markdown(f"""
    <div class="metric-card">
        <h3>COVID-19 Positive</h3>
        <h2>{covid_positive}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    covid_negative = len(df) - covid_positive
    st.markdown(f"""
    <div class="metric-card">
        <h3>COVID-19 Negative</h3>
        <h2>{covid_negative}</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    positivity_rate = (covid_positive / len(df)) * 100
    st.markdown(f"""
    <div class="metric-card">
        <h3>Positivity Rate</h3>
        <h2>{positivity_rate:.1f}%</h2>
    </div>
    """, unsafe_allow_html=True)

# --- 2. Analyse descriptive ---
st.markdown("---")
st.header("📊 Descriptive Analysis")

# Sélecteur de layout
tab1, tab2, tab3, tab4 = st.tabs(["Dataset Overview", "Feature Analysis", "COVID-19 Distribution", "Textual Analysis"])

with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("Dataset Information")
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.write("**Shape:**", df.shape)
        st.write("**Columns:**", list(df.columns))
    
    with col_info2:
        st.write("**Missing Values:**")
        missing_data = df.isnull().sum()
        for col, missing in missing_data.items():
            if missing > 0:
                st.write(f"- {col}: {missing} ({missing/len(df)*100:.1f}%)")

with tab2:
    st.subheader("Feature Distribution")
    feature = st.selectbox("Select a feature to visualize", df.columns[:-1])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Graphique de distribution
    if df[feature].dtype in ['int64', 'float64'] and df[feature].nunique() > 10:
        sns.histplot(data=df, x=feature, hue='COVID-19', ax=ax1, kde=True)
        ax1.set_title(f'Distribution of {feature}')
    else:
        value_counts = df[feature].value_counts()
        ax1.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
        ax1.set_title(f'Distribution of {feature}')
    
    # Graphique par statut COVID
    sns.countplot(data=df, x=feature, hue='COVID-19', ax=ax2)
    ax2.set_title(f'{feature} by COVID-19 Status')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)

with tab3:
    st.subheader("COVID-19 Status Distribution")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Graphique simple
    covid_counts = df['COVID-19'].value_counts()
    colors = ['#ff9999', '#66b3ff']
    ax1.pie(covid_counts.values, labels=['Negative', 'Positive'], autopct='%1.1f%%', colors=colors)
    ax1.set_title('COVID-19 Distribution')
    
    # Graphique détaillé
    sns.countplot(data=df, x='COVID-19', ax=ax2, palette=colors)
    ax2.set_xlabel("COVID-19 Status (0=Negative, 1=Positive)")
    ax2.set_ylabel("Number of Patients")
    ax2.set_title('COVID-19 Cases Count')
    
    # Ajouter les nombres sur les barres
    for i, count in enumerate(covid_counts.values):
        ax2.text(i, count + 5, str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)

with tab4:
    st.subheader("📝 Textual Analysis Report")
    
    # Analyse textuelle détaillée
    st.markdown("### Dataset Summary")
    st.write(f"""
    The dataset contains **{len(df)} patients** with **{len(df.columns)} features**.
    - **COVID-19 Positive Cases**: {covid_positive} ({positivity_rate:.1f}% of total)
    - **COVID-19 Negative Cases**: {covid_negative} ({(100-positivity_rate):.1f}% of total)
    - **Features available**: {', '.join(df.columns[:-1])}
    """)
    
    # Analyse des caractéristiques les plus corrélées
    st.markdown("### Key Insights")
    
    # Calculer les taux de positivité par caractéristique binaire
    binary_features = [col for col in df.columns[:-1] if df[col].nunique() == 2]
    
    if binary_features:
        high_risk_features = []
        for feature in binary_features:
            positivity_rates = df.groupby(feature)['COVID-19'].mean() * 100
            if len(positivity_rates) == 2:
                risk_ratio = positivity_rates[1] / positivity_rates[0] if positivity_rates[0] > 0 else float('inf')
                if risk_ratio > 2.0:  # Seuil pour caractéristiques à haut risque
                    high_risk_features.append((feature, risk_ratio))
        
        if high_risk_features:
            st.write("**🚨 High-Risk Indicators:**")
            for feature, risk_ratio in sorted(high_risk_features, key=lambda x: x[1], reverse=True)[:5]:
                st.write(f"- **{feature}**: {risk_ratio:.1f}x higher COVID-19 risk")
    
    # Distribution par âge si disponible
    age_columns = [col for col in df.columns if 'age' in col.lower() or 'Age' in col]
    if age_columns:
        age_col = age_columns[0]
        st.markdown("### Age Distribution Analysis")
        age_stats = df[age_col].describe()
        st.write(f"""
        - **Average Age**: {age_stats['mean']:.1f} years
        - **Age Range**: {age_stats['min']:.0f} - {age_stats['max']:.0f} years
        - **Median Age**: {age_stats['50%']:.1f} years
        """)

# --- 3. Modèle de prédiction ---
st.markdown("---")
st.header("🤖 Machine Learning Model")

model_tab1, model_tab2 = st.tabs(["Model Training", "Feature Importance"])

with model_tab1:
    st.subheader("Train Random Forest Classifier")
    
    if st.button("🚀 Train & Evaluate Model", type="primary"):
        with st.spinner("Training model... This may take a few moments."):
            # Séparer features / target
            X = df.drop(columns=['COVID-19'])
            y = df['COVID-19']

            # Stocker dans session_state pour réutilisation
            st.session_state.X = X
            st.session_state.y = y

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            
            # Afficher les résultats
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                st.metric("Model Accuracy", f"{accuracy:.2%}")
                st.metric("Test Set Size", len(X_test))
                st.metric("Training Set Size", len(X_train))
            
            with col_result2:
                st.metric("Positive Cases in Test", f"{y_test.sum()}/{len(y_test)}")
                st.metric("Negative Cases in Test", f"{len(y_test)-y_test.sum()}/{len(y_test)}")
            
            # Matrice de confusion
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Predicted Negative', 'Predicted Positive'],
                       yticklabels=['Actual Negative', 'Actual Positive'])
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)
            
            # Rapport de classification
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)
            
            # Sauvegarder le modèle
            joblib.dump(model, 'covid19_model.pkl')
            st.success("✅ Model trained and saved as 'covid19_model.pkl'")

with model_tab2:
    st.subheader("Feature Importance Analysis")
    
    try:
        model = joblib.load('covid19_model.pkl')
        
        # Utiliser X depuis session_state ou le recréer si nécessaire
        if st.session_state.X is not None:
            X = st.session_state.X
        else:
            X = df.drop(columns=['COVID-19'])
        
        importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        # Top 10 des features les plus importantes
        top_features = importances.head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=top_features, x='Importance', y='Feature', palette='viridis')
        ax.set_title('Top 10 Most Important Features')
        ax.set_xlabel('Feature Importance Score')
        st.pyplot(fig)
        
        # Affichage détaillé
        st.write("**Detailed Feature Importances:**")
        for idx, row in top_features.iterrows():
            st.markdown(f"""
            <div class="feature-importance">
                <strong>{row['Feature']}</strong>: {row['Importance']:.3f}
            </div>
            """, unsafe_allow_html=True)
            
    except FileNotFoundError:
        st.info("Please train the model first to see feature importances.")
    except Exception as e:
        st.error(f"Error loading model: {e}")

# --- 4. Prédiction d'un nouveau patient ---
st.markdown("---")
st.header("🎯 COVID-19 Prediction for New Patient")

pred_tab1, pred_tab2 = st.tabs(["Manual Input", "Batch Prediction"])

with pred_tab1:
    st.subheader("Enter Patient Information")
    
    # Layout en colonnes pour l'input
    col_input1, col_input2, col_input3 = st.columns(3)
    
    with col_input1:
        st.markdown("#### 🩺 Symptoms")
        Breathing_Problem = st.selectbox("Breathing Problem", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        Fever = st.selectbox("Fever", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        Dry_Cough = st.selectbox("Dry Cough", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        Sore_throat = st.selectbox("Sore Throat", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        Headache = st.selectbox("Headache", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        Fatigue = st.selectbox("Fatigue", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    
    with col_input2:
        st.markdown("#### 💊 Chronic Conditions")
        Asthma = st.selectbox("Asthma", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        Chronic_Lung_Disease = st.selectbox("Chronic Lung Disease", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        Heart_Disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        Diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        Hyper_Tension = st.selectbox("Hyper Tension", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    
    with col_input3:
        st.markdown("#### 🚨 Exposure & Prevention")
        Abroad_travel = st.selectbox("Abroad Travel", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        Contact_with_COVID_Patient = st.selectbox("Contact with COVID Patient", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        Attended_Large_Gathering = st.selectbox("Attended Large Gathering", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        Visited_Public_Exposed_Places = st.selectbox("Visited Public Exposed Places", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        Wearing_Masks = st.selectbox("Wearing Masks", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    
    # Préparer les données pour la prédiction
    new_patient = pd.DataFrame([{
        'Breathing_Problem': Breathing_Problem,
        'Fever': Fever,
        'Dry_Cough': Dry_Cough,
        'Sore_throat': Sore_throat,
        'Running_Nose': 0,  # Valeur par défaut
        'Asthma': Asthma,
        'Chronic_Lung_Disease': Chronic_Lung_Disease,
        'Headache': Headache,
        'Heart_Disease': Heart_Disease,
        'Diabetes': Diabetes,
        'Hyper_Tension': Hyper_Tension,
        'Fatigue': Fatigue,
        'Gastrointestinal': 0,  # Valeur par défaut
        'Abroad_travel': Abroad_travel,
        'Contact_with_COVID_Patient': Contact_with_COVID_Patient,
        'Attended_Large_Gathering': Attended_Large_Gathering,
        'Visited_Public_Exposed_Places': Visited_Public_Exposed_Places,
        'Family_working_in_Public_Exposed_Places': 0,  # Valeur par défaut
        'Wearing_Masks': Wearing_Masks,
        'Sanitization_from_Market': 0  # Valeur par défaut
    }])
    
    # S'assurer que les colonnes correspondent à l'entraînement
    try:
        model = joblib.load('covid19_model.pkl')
        # Réorganiser les colonnes pour correspondre à l'entraînement
        new_patient = new_patient.reindex(columns=model.feature_names_in_, fill_value=0)
    except:
        pass
    
    if st.button("🔍 Predict COVID-19 Status", type="primary"):
        try:
            model = joblib.load('covid19_model.pkl')
            prediction = model.predict(new_patient)[0]
            prediction_proba = model.predict_proba(new_patient)[0]
            
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-positive">
                    <h2>🚨 COVID-19 POSITIVE</h2>
                    <p>Probability: {prediction_proba[1]:.1%}</p>
                    <p>Recommendation: Please consult a healthcare professional immediately.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-negative">
                    <h2>✅ COVID-19 NEGATIVE</h2>
                    <p>Probability: {prediction_proba[0]:.1%}</p>
                    <p>Recommendation: Continue following safety protocols.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Afficher les probabilités détaillées
            st.subheader("Prediction Probabilities")
            prob_df = pd.DataFrame({
                'Status': ['Negative', 'Positive'],
                'Probability': [prediction_proba[0], prediction_proba[1]]
            })
            st.dataframe(prob_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error in prediction: {e}")
            st.info("Please make sure the model has been trained first.")

with pred_tab2:
    st.subheader("Batch Prediction")
    st.info("Upload a CSV file with patient data for batch predictions")
    
    batch_file = st.file_uploader("Upload patient data CSV", type=["csv"], key="batch")
    
    if batch_file is not None:
        batch_df = pd.read_csv(batch_file)
        batch_df.columns = batch_df.columns.str.strip().str.replace(' ', '_')
        batch_df = batch_df.replace({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})
        
        st.write("**Uploaded Data Preview:**")
        st.dataframe(batch_df.head())
        
        if st.button("🔍 Run Batch Prediction"):
            try:
                model = joblib.load('covid19_model.pkl')
                
                # Préparer les données
                batch_data = batch_df.reindex(columns=model.feature_names_in_, fill_value=0)
                predictions = model.predict(batch_data)
                probabilities = model.predict_proba(batch_data)
                
                # Ajouter les résultats au DataFrame
                results_df = batch_df.copy()
                results_df['COVID-19_Prediction'] = predictions
                results_df['COVID-19_Probability_Negative'] = probabilities[:, 0]
                results_df['COVID-19_Probability_Positive'] = probabilities[:, 1]
                
                st.success(f"✅ Predictions completed for {len(results_df)} patients")
                
                # Statistiques des prédictions
                positive_count = predictions.sum()
                negative_count = len(predictions) - positive_count
                
                col_stat1, col_stat2 = st.columns(2)
                with col_stat1:
                    st.metric("Positive Predictions", positive_count)
                with col_stat2:
                    st.metric("Negative Predictions", negative_count)
                
                # Afficher les résultats
                st.subheader("Prediction Results")
                st.dataframe(results_df, use_container_width=True)
                
                # Option de téléchargement
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions",
                    data=csv,
                    file_name=f"covid_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"❌ Error in batch prediction: {e}")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>COVID-19 Analysis Dashboard • Built with Streamlit</p>
    <p><small>For educational and research purposes only</small></p>
</div>
""", unsafe_allow_html=True)