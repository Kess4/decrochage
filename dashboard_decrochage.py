"""
Tableau de bord interactif pour la prédiction du risque de décrochage
EPITECH Bordeaux - POC pour accompagnateurs et enseignants
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import json
import requests
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Prédiction Décrochage - EPITECH Bordeaux",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé amélioré
st.markdown("""
<style>
    .alert-critical {
        background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        font-weight: bold;
        font-size: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .alert-high {
        background: linear-gradient(135deg, #ff8800 0%, #cc6600 100%);
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .student-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .student-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .student-card-critical {
        border-left: 6px solid #ff0000;
        background: linear-gradient(90deg, #ffe6e6 0%, #ffffff 100%);
    }
    .student-card-high {
        border-left: 6px solid #ff8800;
        background: linear-gradient(90deg, #fff4e6 0%, #ffffff 100%);
    }
    .student-card-moderate {
        border-left: 6px solid #ffbb00;
        background: linear-gradient(90deg, #fff9e6 0%, #ffffff 100%);
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .big-number {
        font-size: 36px;
        font-weight: bold;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Charge le dataset"""
    try:
        df = pd.read_csv('dataset_epitech_bordeaux_decrochage.csv', encoding='utf-8-sig')
        return df, None
    except FileNotFoundError:
        return None, "Fichier dataset_epitech_bordeaux_decrochage.csv non trouvé. Veuillez d'abord générer le dataset."

@st.cache_resource
def load_models():
    """Charge les modèles ML"""
    try:
        model_decrochage = joblib.load('modele_decrochage.pkl')
        model_risque = joblib.load('modele_risque_score.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        feature_names = joblib.load('feature_names.pkl')
        
        try:
            scaler = joblib.load('scaler.pkl')
        except FileNotFoundError:
            scaler = None
        
        return (model_decrochage, model_risque, label_encoders, feature_names, scaler), None
    except FileNotFoundError as e:
        return None, "Modèles ML non trouvés. Veuillez d'abord exécuter le notebook modele_prediction_decrochage.ipynb"

def prepare_features(df_row, label_encoders, feature_names):
    """Prépare les features pour la prédiction"""
    features = {}
    
    for col in feature_names:
        if col in df_row.index:
            value = df_row[col]
            if col in label_encoders:
                try:
                    features[col] = label_encoders[col].transform([value])[0]
                except ValueError:
                    features[col] = 0
            else:
                features[col] = value
        else:
            features[col] = 0
    
    feature_df = pd.DataFrame([features])[feature_names]
    return feature_df

def predict_risk(df_row, model_decrochage, model_risque, label_encoders, feature_names, scaler=None):
    """Prédit le risque de décrochage et le score de risque"""
    feature_df = prepare_features(df_row, label_encoders, feature_names)
    
    if scaler is not None:
        feature_scaled = scaler.transform(feature_df)
        decrochage_pred = model_decrochage.predict(feature_scaled)[0]
        decrochage_proba = model_decrochage.predict_proba(feature_scaled)[0][1]
    else:
        decrochage_pred = model_decrochage.predict(feature_df)[0]
        decrochage_proba = model_decrochage.predict_proba(feature_df)[0][1]
    
    risque_score = model_risque.predict(feature_df)[0]
    risque_score = max(0, min(1, risque_score))
    
    return decrochage_pred, decrochage_proba, risque_score

def get_risk_level(risque_score):
    """Détermine le niveau de risque"""
    if risque_score >= 0.7:
        return "Critique", "🔴", "#ff0000", "rgba(255, 0, 0, 0.25)"
    elif risque_score >= 0.5:
        return "Élevé", "🟠", "#ff8800", "rgba(255, 136, 0, 0.25)"
    elif risque_score >= 0.3:
        return "Modéré", "🟡", "#ffbb00", "rgba(255, 187, 0, 0.25)"
    else:
        return "Faible", "🟢", "#00cc00", "rgba(0, 204, 0, 0.25)"

def get_recommendations(df_row, risque_score, decrochage_pred):
    """Génère des recommandations personnalisées"""
    recommendations = []
    
    if df_row['note_moyenne'] < 10:
        recommendations.append("📚 **Soutien académique** : Proposer un accompagnement renforcé en cours")
    
    if df_row['taux_absences'] > 15:
        recommendations.append("📞 **Contact urgent** : Contacter l'étudiant pour comprendre les absences")
    
    if df_row['participation_projets'] < 0.3:
        recommendations.append("💼 **Projets** : Organiser un entretien pour identifier les difficultés sur les projets")
    
    if df_row['nb_projets_en_retard'] > 2:
        recommendations.append("⏰ **Gestion du temps** : Mettre en place un accompagnement sur la planification")
    
    if df_row['nb_rdv_pedagogique'] == 0 and df_row['annee_etude'] == 1:
        recommendations.append("🤝 **Premier contact** : Proposer un rendez-vous pédagogique d'accueil")
    
    if df_row['satisfaction_formation'] < 0.4:
        recommendations.append("💬 **Enquête** : Enquêter sur les causes de l'insatisfaction")
    
    if risque_score >= 0.7:
        recommendations.insert(0, "🚨 **ACTION IMMÉDIATE** : Contacter l'étudiant en urgence et organiser un entretien")
    
    if not recommendations:
        recommendations.append("✅ Profil stable, continuer le suivi régulier")
    
    return recommendations

# Chargement des données
df, error_data = load_data()
if df is None:
    st.error(f"❌ {error_data}")
    st.stop()

models_result, error_models = load_models()
if models_result is None:
    st.error(f"❌ {error_models}")
    st.stop()

model_decrochage, model_risque, label_encoders, feature_names, scaler = models_result

# Prédictions pour tous les étudiants
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
    with st.spinner("Calcul des prédictions en cours..."):
        for idx, row in df.iterrows():
            decrochage_pred, decrochage_proba, risque_score = predict_risk(
                row, model_decrochage, model_risque, label_encoders, feature_names, scaler
            )
            st.session_state.predictions.append({
                'id_etudiant': row['id_etudiant'],
                'decrochage_pred': decrochage_pred,
                'decrochage_proba': decrochage_proba,
                'risque_score': risque_score
            })

# Création d'un DataFrame avec les prédictions
predictions_df = pd.DataFrame(st.session_state.predictions)
df_with_predictions = df.merge(predictions_df, on='id_etudiant', how='left')

# Sidebar avec filtres
st.sidebar.header("🔍 Filtres")
st.sidebar.markdown("---")

programmes = ['Tous'] + list(df['programme'].unique())
selected_programme = st.sidebar.selectbox("Programme", programmes)

annees = ['Toutes'] + sorted(df['annee_etude'].unique().tolist())
selected_annee = st.sidebar.selectbox("Année d'étude", annees)

niveau_risque = st.sidebar.selectbox(
    "Niveau de risque",
    ['Tous', 'Critique (≥70%)', 'Élevé (50-70%)', 'Modéré (30-50%)', 'Faible (<30%)']
)

# Application des filtres
df_filtered = df_with_predictions.copy()

if selected_programme != 'Tous':
    df_filtered = df_filtered[df_filtered['programme'] == selected_programme]

if selected_annee != 'Toutes':
    df_filtered = df_filtered[df_filtered['annee_etude'] == selected_annee]

if niveau_risque == 'Critique (≥70%)':
    df_filtered = df_filtered[df_filtered['risque_score'] >= 0.7]
elif niveau_risque == 'Élevé (50-70%)':
    df_filtered = df_filtered[(df_filtered['risque_score'] >= 0.5) & (df_filtered['risque_score'] < 0.7)]
elif niveau_risque == 'Modéré (30-50%)':
    df_filtered = df_filtered[(df_filtered['risque_score'] >= 0.3) & (df_filtered['risque_score'] < 0.5)]
elif niveau_risque == 'Faible (<30%)':
    df_filtered = df_filtered[df_filtered['risque_score'] < 0.3]

# Calcul des statistiques
total_etudiants = len(df_filtered)
nb_decrochage = df_filtered['decrochage_pred'].sum()
nb_critique = len(df_filtered[df_filtered['risque_score'] >= 0.7])
nb_eleve = len(df_filtered[(df_filtered['risque_score'] >= 0.5) & (df_filtered['risque_score'] < 0.7)])
nb_modere = len(df_filtered[(df_filtered['risque_score'] >= 0.3) & (df_filtered['risque_score'] < 0.5)])
risque_moyen = df_filtered['risque_score'].mean() * 100 if total_etudiants > 0 else 0

# En-tête principal
st.title("🎓 Tableau de Bord - Prédiction du Risque de Décrochage")
st.markdown("**EPITECH Bordeaux** - Outil d'aide à la décision pour enseignants et responsables pédagogiques")

# ALERTES CRITIQUES en haut
if nb_critique > 0:
    st.markdown(f"""
    <div class="alert-critical">
        🚨 ALERTE CRITIQUE : {nb_critique} étudiant(s) nécessitent une action immédiate !
    </div>
    """, unsafe_allow_html=True)

if nb_eleve > 0:
    st.markdown(f"""
    <div class="alert-high">
        ⚠️ {nb_eleve} étudiant(s) présentent un risque élevé de décrochage
    </div>
    """, unsafe_allow_html=True)

# ONGLETS
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Vue d'ensemble", "👥 Étudiants à risque", "📈 Analyses", "🔍 Détail étudiant", "🔔 Alertes"])

# ONGLET 1: Vue d'ensemble
with tab1:
    st.header("Vue d'ensemble")
    
    # KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total étudiants", total_etudiants)
    
    with col2:
        st.metric("Risque de décrochage", f"{nb_decrochage}", 
                 delta=f"{nb_decrochage/total_etudiants*100:.1f}%" if total_etudiants > 0 else "0%",
                 delta_color="inverse")
    
    with col3:
        st.metric("🔴 Critiques", nb_critique, 
                 delta=f"{nb_critique/total_etudiants*100:.1f}%" if total_etudiants > 0 else "0%",
                 delta_color="inverse")
    
    with col4:
        st.metric("🟠 Élevé", nb_eleve,
                 delta=f"{nb_eleve/total_etudiants*100:.1f}%" if total_etudiants > 0 else "0%",
                 delta_color="inverse")
    
    with col5:
        st.metric("Risque moyen", f"{risque_moyen:.1f}%")
    
    st.markdown("---")
    
    # Répartition par niveau de risque
    col1, col2 = st.columns(2)
    
    with col1:
        # Graphique en camembert - Répartition des risques
        risk_distribution = {
            'Critique (≥70%)': nb_critique,
            'Élevé (50-70%)': nb_eleve,
            'Modéré (30-50%)': nb_modere,
            'Faible (<30%)': total_etudiants - nb_critique - nb_eleve - nb_modere
        }
        
        fig_pie = px.pie(
            values=list(risk_distribution.values()),
            names=list(risk_distribution.keys()),
            title="Répartition par niveau de risque",
            color_discrete_sequence=['#d32f2f', '#f57c00', '#fbc02d', '#66bb6a']  # Couleurs plus douces
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Graphique barres - Risque par programme
        risque_prog = df_filtered.groupby('programme')['risque_score'].agg(['mean', 'count']).reset_index()
        risque_prog.columns = ['Programme', 'Risque moyen', 'Nombre']
        
        fig_bar = px.bar(
            risque_prog,
            x='Programme',
            y='Risque moyen',
            title='Risque moyen par programme',
            labels={'Risque moyen': 'Score de risque moyen', 'Programme': 'Programme'},
            color='Risque moyen',
            color_continuous_scale='Reds',
            text='Nombre'
        )
        fig_bar.update_traces(texttemplate='%{text} étudiants', textposition='outside')
        fig_bar.update_layout(yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Graphique ligne - Évolution par année
    st.markdown("### Évolution du risque par année d'étude")
    risque_annee = df_filtered.groupby('annee_etude')['risque_score'].agg(['mean', 'count']).reset_index()
    risque_annee.columns = ['Année', 'Risque moyen', 'Nombre']
    
    fig_line = px.line(
        risque_annee,
        x='Année',
        y='Risque moyen',
        title='Risque moyen par année d\'étude',
        labels={'Risque moyen': 'Score de risque moyen', 'Année': 'Année d\'étude'},
        markers=True,
        text='Nombre'
    )
    fig_line.update_traces(
        line_color='#ff4444',
        marker_color='#ff4444',
        line_width=3,
        texttemplate='%{text} étudiants',
        textposition='top center'
    )
    fig_line.update_layout(yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig_line, use_container_width=True)

# ONGLET 2: Étudiants à risque
with tab2:
    st.header("Étudiants à risque")
    
    # Filtrer uniquement les étudiants à risque (exclure les profils stables < 30%)
    df_risque = df_filtered[df_filtered['risque_score'] >= 0.3].copy()
    df_sorted = df_risque.sort_values('risque_score', ascending=False)
    
    if len(df_sorted) > 0:
        # Statistiques rapides
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔴 Critiques", len(df_sorted[df_sorted['risque_score'] >= 0.7]))
        with col2:
            st.metric("🟠 Élevé", len(df_sorted[(df_sorted['risque_score'] >= 0.5) & (df_sorted['risque_score'] < 0.7)]))
        with col3:
            st.metric("🟡 Modéré", len(df_sorted[(df_sorted['risque_score'] >= 0.3) & (df_sorted['risque_score'] < 0.5)]))
        
        st.markdown("---")
        
        # Tableau simplifié avec mise en forme conditionnelle
        st.subheader("📋 Liste des étudiants (triée par niveau de risque)")
        
        # Préparer les données pour l'affichage
        display_cols = ['id_etudiant', 'programme', 'annee_etude', 'note_moyenne', 
                       'taux_absences', 'participation_projets', 'nb_projets_en_retard',
                       'risque_score', 'decrochage_pred']
        
        df_display = df_sorted[display_cols].copy()
        
        # Ajouter une colonne pour le niveau de risque (pour le style)
        def get_risk_category(score):
            if score >= 0.7:
                return "Critique"
            elif score >= 0.5:
                return "Élevé"
            elif score >= 0.3:
                return "Modéré"
            else:
                return "Faible"
        
        df_display['Niveau'] = df_display['risque_score'].apply(get_risk_category)
        df_display['Score %'] = df_display['risque_score'].apply(lambda x: f"{x*100:.1f}%")
        df_display['Décrochage'] = df_display['decrochage_pred'].apply(lambda x: "⚠️ Oui" if x == 1 else "✅ Non")
        df_display['Participation'] = df_display['participation_projets'].apply(lambda x: f"{x*100:.0f}%")
        
        # Réorganiser les colonnes
        df_display = df_display[['id_etudiant', 'Niveau', 'programme', 'annee_etude', 
                                'note_moyenne', 'taux_absences', 'Participation', 
                                'nb_projets_en_retard', 'Score %', 'Décrochage']]
        df_display.columns = ['ID Étudiant', 'Niveau Risque', 'Programme', 'Année', 
                             'Note Moyenne', 'Taux Absences', 'Participation Projets', 
                             'Projets Retard', 'Score Risque', 'Décrochage']
        
        # Afficher le tableau avec style
        st.dataframe(
            df_display,
            use_container_width=True,
            height=600,
            hide_index=True
        )
        
        # Légende
        st.markdown("""
        <div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin-top: 10px;'>
            <strong>Légende :</strong><br>
            🔴 <strong>Critique</strong> (≥70%) : Action immédiate requise | 
            🟠 <strong>Élevé</strong> (50-70%) : Suivi renforcé recommandé | 
            🟡 <strong>Modéré</strong> (30-50%) : Surveillance continue | 
            🟢 <strong>Faible</strong> (<30%) : Profil stable
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Aucun étudiant ne correspond aux filtres sélectionnés.")

# ONGLET 3: Analyses
with tab3:
    st.header("Analyses approfondies")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution du risque
        fig_hist = px.histogram(
            df_filtered,
            x='risque_score',
            nbins=30,
            title='Distribution du score de risque',
            labels={'risque_score': 'Score de risque', 'count': 'Nombre d\'étudiants'},
            color_discrete_sequence=['#ff4444']
        )
        fig_hist.add_vline(x=0.7, line_dash="dash", line_color="red", 
                          annotation_text="Seuil critique (70%)", annotation_position="top")
        fig_hist.add_vline(x=0.5, line_dash="dash", line_color="orange", 
                          annotation_text="Seuil élevé (50%)", annotation_position="top")
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Scatter plot Note vs Risque
        fig_scatter = px.scatter(
            df_filtered,
            x='note_moyenne',
            y='risque_score',
            color='risque_score',
            size='taux_absences',
            hover_data=['id_etudiant', 'programme', 'annee_etude'],
            title='Note moyenne vs Score de risque',
            labels={'note_moyenne': 'Note moyenne (/20)', 'risque_score': 'Score de risque'},
            color_continuous_scale='RdYlGn_r',
            size_max=20
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Matrice de corrélation
    st.subheader("Corrélations entre facteurs de risque")
    numeric_cols = ['note_moyenne', 'note_projet', 'taux_absences', 'participation_projets',
                   'participation_cours', 'nb_projets_en_retard', 'satisfaction_formation',
                   'risque_score']
    
    corr_matrix = df_filtered[numeric_cols].corr()
    
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Matrice de corrélation",
        color_continuous_scale='RdBu',
        labels=dict(color="Corrélation")
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# Fonction pour envoyer des alertes
def send_email_alert(to_email, subject, body, smtp_config=None):
    """Envoie un email d'alerte"""
    if smtp_config is None:
        return False, "Configuration SMTP non définie"
    
    try:
        msg = MIMEMultipart()
        msg['From'] = smtp_config['from_email']
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'html'))
        
        port = int(smtp_config['smtp_port'])
        
        # Gestion SSL (port 465) vs TLS (port 587)
        if port == 465:
            # Connexion SSL
            server = smtplib.SMTP_SSL(smtp_config['smtp_server'], port)
        else:
            # Connexion TLS (port 587 ou autres)
            server = smtplib.SMTP(smtp_config['smtp_server'], port)
            server.starttls()
        
        server.login(smtp_config['from_email'], smtp_config['password'])
        server.send_message(msg)
        server.quit()
        
        return True, "Email envoyé avec succès"
    except smtplib.SMTPAuthenticationError:
        return False, "Erreur d'authentification : Vérifiez votre email et mot de passe"
    except smtplib.SMTPConnectError as e:
        return False, f"Erreur de connexion au serveur SMTP : {str(e)}. Vérifiez le serveur ({smtp_config['smtp_server']}) et le port ({port})"
    except (smtplib.SMTPServerDisconnected, ConnectionError) as e:
        return False, f"Connexion interrompue : {str(e)}. Le serveur a fermé la connexion. Vérifiez votre connexion réseau et les paramètres SMTP."
    except Exception as e:
        error_msg = str(e)
        return False, f"Erreur lors de l'envoi : {error_msg}"

def send_teams_webhook(webhook_url, title, message, color="FF0000", is_workflow=False, message_data=None):
    """Envoie une alerte Teams via webhook ou workflow HTTP avec timeout"""
    try:
        if is_workflow:
            # Format pour les workflows HTTP de Teams
            if isinstance(message, dict):
                # Si c'est un dict, convertir en texte
                msg_text = f"{title}\n\n"
                msg_text += f"Date : {message.get('date', '')}\n"
                msg_text += f"Nombre d'étudiants concernés : {message.get('nombre_etudiants', 0)}\n\n"
                msg_text += "Résumé :\n"
                resume = message.get('resume', {})
                msg_text += f"- Profils critiques (≥70%) : {resume.get('critiques', 0)}\n"
                msg_text += f"- Risque élevé (50-70%) : {resume.get('eleve', 0)}\n"
                msg_text += f"- Risque modéré (30-50%) : {resume.get('modere', 0)}\n\n"
                msg_text += "Liste des étudiants :\n"
                for etudiant in message.get('etudiants', []):
                    msg_text += f"- {etudiant['id']} ({etudiant['programme']}, Année {etudiant['annee']}) : {etudiant['emoji']} {etudiant['score']:.1f}% - Note: {etudiant['note']:.1f}/20\n"
                if message.get('etudiants_restants', 0) > 0:
                    msg_text += f"... et {message['etudiants_restants']} autre(s) étudiant(s)\n"
                msg_text += "\nAction recommandée : Contacter les étudiants concernés et organiser des entretiens pédagogiques."
                message = msg_text
            
            payload = {
                "title": title,
                "message": message,
                "color": color
            }
        else:
            # Format MessageCard enrichi pour les webhooks classiques
            if isinstance(message, dict):
                # Construire le payload directement depuis les données structurées
                sections = [{
                    "activityTitle": title,
                    "activitySubtitle": f"Date : {message.get('date', '')}",
                    "facts": [
                        {"name": "Nombre d'étudiants concernés :", "value": str(message.get('nombre_etudiants', 0))}
                    ]
                }]
                
                # Section Résumé
                resume = message.get('resume', {})
                sections.append({
                    "title": "Résumé :",
                    "facts": [
                        {"name": "🔴 Profils critiques (≥70%)", "value": str(resume.get('critiques', 0))},
                        {"name": "🟠 Risque élevé (50-70%)", "value": str(resume.get('eleve', 0))},
                        {"name": "🟡 Risque modéré (30-50%)", "value": str(resume.get('modere', 0))}
                    ]
                })
                
                # Section Liste des étudiants
                # Utiliser des facts pour créer un format structuré de type tableau
                # Teams MessageCard ne supporte pas les tableaux HTML, mais les facts créent un format similaire
                etudiants_facts = []
                
                # Limiter à 15 étudiants pour éviter un message trop long
                etudiants_affiches = message.get('etudiants', [])[:15]
                
                for etudiant in etudiants_affiches:
                    # Créer un fait pour chaque étudiant avec toutes les informations
                    etudiant_info = f"{etudiant['id']} | {etudiant['programme']} | Année {etudiant['annee']} | {etudiant['emoji']} {etudiant['score']:.1f}% | Note: {etudiant['note']:.1f}/20"
                    etudiants_facts.append({
                        "name": f"• {etudiant['id']}",
                        "value": f"{etudiant['programme']} - Année {etudiant['annee']} | {etudiant['emoji']} {etudiant['score']:.1f}% | Note: {etudiant['note']:.1f}/20"
                    })
                
                # Si il y a des étudiants restants, ajouter une ligne d'information
                if message.get('etudiants_restants', 0) > 0:
                    etudiants_facts.append({
                        "name": "...",
                        "value": f"{message['etudiants_restants']} autre(s) étudiant(s)"
                    })
                
                sections.append({
                    "title": "Liste des étudiants :",
                    "facts": etudiants_facts
                })
                
                # Section Action recommandée
                sections.append({
                    "title": "Action recommandée :",
                    "text": "Contacter les étudiants concernés et organiser des entretiens pédagogiques."
                })
                
                payload = {
                    "@type": "MessageCard",
                    "@context": "http://schema.org/extensions",
                    "themeColor": color,
                    "summary": title,
                    "sections": sections
                }
            else:
                # Parser le message texte pour créer des sections structurées
                lines = message.strip().split('\n')
                
                # Section principale avec titre et date
                main_section = {
                    "activityTitle": title,
                    "activitySubtitle": f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}",
                    "facts": []
                }
                
                # Sections supplémentaires
                sections = [main_section]
                current_section = None
                students_text = ""
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Détecter les informations clé-valeur
                    if ':' in line and not line.startswith('•') and not line.startswith('-') and not line.startswith('...'):
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value = parts[1].strip()
                            
                            # Si c'est "Liste des étudiants", créer une nouvelle section
                            if "Liste des étudiants" in key:
                                current_section = {
                                    "title": "Liste des étudiants :",
                                    "text": ""
                                }
                                sections.append(current_section)
                            # Si c'est "Action recommandée", créer une nouvelle section
                            elif "Action recommandée" in key:
                                sections.append({
                                    "title": "Action recommandée :",
                                    "text": value
                                })
                            # Sinon, ajouter comme fait
                            else:
                                main_section["facts"].append({"name": key, "value": value})
                    
                    # Détecter les éléments de liste (étudiants)
                    elif line.startswith('•') or line.startswith('-'):
                        if current_section:
                            current_section["text"] += line[1:].strip() + "\n"
                        else:
                            students_text += line[1:].strip() + "\n"
                    
                    # Détecter les lignes de continuation
                    elif line.startswith('...'):
                        if current_section:
                            current_section["text"] += line + "\n"
                        else:
                            students_text += line + "\n"
                
                # Si on a collecté des étudiants mais pas de section, créer une section
                if students_text and not any(s.get("title") == "Liste des étudiants" for s in sections):
                    sections.append({
                        "title": "Liste des étudiants :",
                        "text": students_text.strip()
                    })
                
                # Si on a une section étudiants avec du texte, l'utiliser
                for section in sections:
                    if section.get("title") == "Liste des étudiants" and not section.get("text") and students_text:
                        section["text"] = students_text.strip()
                
                # Si aucune structure n'a été trouvée, utiliser le message tel quel
                if len(main_section["facts"]) == 0 and len(sections) == 1:
                    main_section["text"] = message
                
                payload = {
                    "@type": "MessageCard",
                    "@context": "http://schema.org/extensions",
                    "themeColor": color,
                    "summary": title,
                    "sections": sections
                }
        
        # Timeout de 10 secondes pour éviter les blocages
        response = requests.post(webhook_url, json=payload, timeout=10)
        if response.status_code in [200, 201, 202, 204]:
            return True, "Alerte Teams envoyée avec succès"
        else:
            return False, f"Erreur HTTP {response.status_code}: {response.text[:200]}"
    except requests.exceptions.Timeout:
        return False, "Timeout : La requête a pris trop de temps"
    except requests.exceptions.ConnectionError:
        return False, "Erreur de connexion : Vérifiez votre connexion internet"
    except Exception as e:
        return False, f"Erreur lors de l'envoi : {str(e)}"

# ONGLET 4: Détail étudiant
with tab4:
    st.header("Détail par étudiant")
    
    if len(df_filtered) > 0:
        # Sélection de l'étudiant
        selected_student = st.selectbox(
            "Sélectionner un étudiant",
            df_filtered['id_etudiant'].tolist(),
            key="student_selector"
        )
        
        if selected_student:
            student_data = df_filtered[df_filtered['id_etudiant'] == selected_student].iloc[0]
            risk_level, emoji, color, fillcolor_rgba = get_risk_level(student_data['risque_score'])
            
            # En-tête avec niveau de risque
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, {color}15 0%, {color}05 100%);
                        padding: 20px; border-radius: 10px; border-left: 6px solid {color}; margin-bottom: 20px;'>
                <h2>{emoji} {selected_student}</h2>
                <p style='font-size: 18px;'><strong>Niveau de risque : {risk_level}</strong> | 
                Score : {student_data['risque_score']*100:.1f}% | 
                Probabilité de décrochage : {student_data['decrochage_proba']*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📚 Informations académiques")
                st.write(f"**Programme :** {student_data['programme']}")
                st.write(f"**Année :** {int(student_data['annee_etude'])}")
                st.write(f"**Classe :** {student_data['classe']}")
                st.write(f"**Note moyenne :** {student_data['note_moyenne']:.1f}/20")
                st.write(f"**Note programmation :** {student_data['note_programmation']:.1f}/20")
                st.write(f"**Note projet :** {student_data['note_projet']:.1f}/20")
                st.write(f"**Note innovation :** {student_data['note_innovation']:.1f}/20")
            
            with col2:
                st.markdown("### 📊 Engagement et comportement")
                st.write(f"**Taux d'absences :** {student_data['taux_absences']:.1f}%")
                st.write(f"**Nombre d'absences :** {int(student_data['nb_absences'])}")
                st.write(f"**Nombre de retards :** {int(student_data['nb_retards'])}")
                st.write(f"**Participation projets :** {student_data['participation_projets']*100:.1f}%")
                st.write(f"**Participation cours :** {student_data['participation_cours']*100:.1f}%")
                st.write(f"**Projets terminés :** {int(student_data['nb_projets_termines'])}")
                st.write(f"**Projets en retard :** {int(student_data['nb_projets_en_retard'])}")
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown("### 🤝 Interactions")
                st.write(f"**RDV pédagogiques :** {int(student_data['nb_rdv_pedagogique'])}")
                st.write(f"**Demandes d'aide :** {int(student_data['nb_demandes_aide'])}")
                st.write(f"**Rappels discipline :** {int(student_data['nb_rappel_discipline'])}")
                st.write(f"**Échecs évaluations :** {int(student_data['nb_echec_evaluation'])}")
            
            with col4:
                st.markdown("### 💡 Autres informations")
                st.write(f"**Temps d'étude/semaine :** {int(student_data['temps_etude_semaine'])}h")
                st.write(f"**Heures cours/semaine :** {int(student_data['nb_heures_cours_semaine'])}h")
                st.write(f"**Satisfaction formation :** {student_data['satisfaction_formation']*100:.1f}%")
                st.write(f"**Boursier :** {'Oui' if student_data['boursier'] == 1 else 'Non'}")
            
            st.markdown("---")
            
            # Recommandations
            st.markdown("### 💡 Recommandations d'action")
            recommendations = get_recommendations(student_data, student_data['risque_score'], student_data['decrochage_pred'])
            
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")
            
            st.markdown("---")
            
            # Graphique radar
            st.markdown("### 📈 Profil de l'étudiant")
            
            categories = ['Notes', 'Assiduité', 'Projets', 'Participation', 'Satisfaction']
            
            values = [
                student_data['note_moyenne'] / 20,
                1 - (student_data['taux_absences'] / 30),
                student_data['participation_projets'],
                student_data['participation_cours'],
                student_data['satisfaction_formation']
            ]
            
            fig_radar = go.Figure()
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=selected_student,
                line_color=color,
                fillcolor=fillcolor_rgba
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Profil multidimensionnel de l'étudiant",
                height=500
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.warning("Aucun étudiant ne correspond aux filtres sélectionnés.")

# ONGLET 5: Alertes
with tab5:
    st.header("🔔 Gestion des alertes")
    st.markdown("Configurez et envoyez des alertes pour les étudiants à risque")
    
    # Section de configuration
    st.subheader("⚙️ Configuration")
    
    alert_type = st.radio(
        "Type d'alerte",
        ["Email", "Microsoft Teams", "Les deux"],
        horizontal=True
    )
    
    # Configuration Email
    if alert_type in ["Email", "Les deux"]:
        st.markdown("### 📧 Configuration Email")
        
        col1, col2 = st.columns(2)
        with col1:
            smtp_server = st.text_input("Serveur SMTP", value="smtp.gmail.com", key="smtp_server")
            smtp_port = st.number_input("Port SMTP", value=587, key="smtp_port")
        with col2:
            from_email = st.text_input("Email expéditeur", key="from_email")
            email_password = st.text_input("Mot de passe", type="password", key="email_password") #itjs blzl hujz gxrc
        
        recipient_email = st.text_input("Email destinataire", key="recipient_email")
    
    # Configuration Teams
    if alert_type in ["Microsoft Teams", "Les deux"]:
        st.markdown("### 💬 Configuration Microsoft Teams")
        
        teams_type = st.radio(
            "Type de connexion Teams",
            ["Webhook", "Workflow HTTP"],
            horizontal=True,
            help="Webhook : URL de webhook classique | Workflow HTTP : URL de workflow Power Automate ou Logic Apps",
            key="teams_type"
        )
        
        teams_webhook = st.text_input(
            f"URL {'du workflow HTTP' if teams_type == 'Workflow HTTP' else 'du webhook'}",
            help="Webhook : Teams > Connexions > Webhooks | Workflow : URL HTTP de votre workflow Power Automate/Logic Apps",
            key="teams_webhook" #https://epitechfr.webhook.office.com/webhookb2/c33efbd1-4180-476f-b9b6-1d41c3350323@901cb4ca-b862-4029-9306-e5cd0f6d9f86/IncomingWebhook/88effca7a7714558b3e854c9c3401cc6/7dc92643-329d-40da-8b1d-31d77fc37970/V2cghjBARSfmKtSC5cbDrsUCP85YkAQKBLeo2vS92Pjx01
        )
    
    st.markdown("---")
    
    # Sélection des étudiants pour alerte
    st.subheader("📋 Sélection des étudiants")
    
    # Filtrer uniquement les étudiants à risque
    df_alertes = df_filtered[df_filtered['risque_score'] >= 0.3].copy()
    df_alertes = df_alertes.sort_values('risque_score', ascending=False)
    
    if len(df_alertes) > 0:
        # Options de sélection
        selection_mode = st.radio(
            "Mode de sélection",
            ["Tous les étudiants à risque", "Critiques uniquement (≥70%)", "Sélection manuelle"],
            horizontal=True
        )
        
        if selection_mode == "Tous les étudiants à risque":
            selected_students = df_alertes.copy()
        elif selection_mode == "Critiques uniquement (≥70%)":
            selected_students = df_alertes[df_alertes['risque_score'] >= 0.7].copy()
        else:
            # Sélection manuelle
            student_list = df_alertes['id_etudiant'].tolist()
            selected_ids = st.multiselect(
                "Sélectionner les étudiants",
                student_list,
                default=student_list[:5] if len(student_list) > 5 else student_list
            )
            selected_students = df_alertes[df_alertes['id_etudiant'].isin(selected_ids)].copy()
        
        if len(selected_students) > 0:
            st.info(f"📌 {len(selected_students)} étudiant(s) sélectionné(s) pour l'alerte")
            
            # Aperçu des étudiants sélectionnés
            with st.expander("👀 Aperçu des étudiants sélectionnés"):
                preview_cols = ['id_etudiant', 'programme', 'annee_etude', 'risque_score', 'note_moyenne']
                preview_df = selected_students[preview_cols].copy()
                preview_df['risque_score'] = preview_df['risque_score'].apply(lambda x: f"{x*100:.1f}%")
                preview_df.columns = ['ID Étudiant', 'Programme', 'Année', 'Score Risque', 'Note Moyenne']
                st.dataframe(preview_df, hide_index=True, use_container_width=True)
            
            st.markdown("---")
            
            # Personnalisation du message
            st.subheader("✍️ Personnalisation du message")
            
            alert_title = st.text_input(
                "Titre de l'alerte",
                value=f"🚨 Alerte Décrochage - {len(selected_students)} étudiant(s) à risque",
                key="alert_title"
            )
            
            # Génération automatique du message
            message_template = st.selectbox(
                "Modèle de message",
                ["Automatique (recommandé)", "Personnalisé"],
                key="message_template"
            )
            
            if message_template == "Automatique (recommandé)":
                # Générer le message automatiquement
                message_body = f"""
                <h2>🚨 Alerte - Étudiants à risque de décrochage</h2>
                <p><strong>Date :</strong> {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
                <p><strong>Nombre d'étudiants concernés :</strong> {len(selected_students)}</p>
                
                <h3>Résumé :</h3>
                <ul>
                    <li>🔴 Profils critiques (≥70%) : {len(selected_students[selected_students['risque_score'] >= 0.7])}</li>
                    <li>🟠 Risque élevé (50-70%) : {len(selected_students[(selected_students['risque_score'] >= 0.5) & (selected_students['risque_score'] < 0.7)])}</li>
                    <li>🟡 Risque modéré (30-50%) : {len(selected_students[(selected_students['risque_score'] >= 0.3) & (selected_students['risque_score'] < 0.5)])}</li>
                </ul>
                
                <h3>Liste des étudiants :</h3>
                <table border="1" style="border-collapse: collapse; width: 100%;">
                    <tr>
                        <th>ID Étudiant</th>
                        <th>Programme</th>
                        <th>Année</th>
                        <th>Score Risque</th>
                        <th>Note Moyenne</th>
                    </tr>
                """
                
                for idx, row in selected_students.iterrows():
                    risk_level, emoji, color, _ = get_risk_level(row['risque_score'])
                    message_body += f"""
                    <tr>
                        <td>{row['id_etudiant']}</td>
                        <td>{row['programme']}</td>
                        <td>{int(row['annee_etude'])}</td>
                        <td style="color: {color}; font-weight: bold;">{emoji} {row['risque_score']*100:.1f}%</td>
                        <td>{row['note_moyenne']:.1f}/20</td>
                    </tr>
                    """
                
                message_body += """
                </table>
                
                <p><strong>Action recommandée :</strong> Contacter les étudiants concernés et organiser des entretiens pédagogiques.</p>
                """
            else:
                message_body = st.text_area(
                    "Message personnalisé (HTML supporté)",
                    height=200,
                    key="custom_message"
                )
            
            # Aperçu du message
            with st.expander("👀 Aperçu du message"):
                st.markdown(message_body, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Programmation des alertes
            st.subheader("⏰ Programmation des alertes")
            
            schedule_alert = st.checkbox("Programmer une alerte récurrente", key="schedule_alert")
            
            if schedule_alert:
                col1, col2 = st.columns(2)
                with col1:
                    frequency = st.selectbox(
                        "Fréquence",
                        ["Quotidienne", "Hebdomadaire", "Mensuelle"],
                        key="alert_frequency"
                    )
                with col2:
                    if frequency == "Quotidienne":
                        schedule_time = st.time_input("Heure d'envoi", key="schedule_time")
                    elif frequency == "Hebdomadaire":
                        schedule_day = st.selectbox("Jour de la semaine", 
                                                   ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi"],
                                                   key="schedule_day")
                        schedule_time = st.time_input("Heure d'envoi", key="schedule_time")
                    else:
                        schedule_day = st.number_input("Jour du mois (1-28)", min_value=1, max_value=28, value=1, key="schedule_day")
                        schedule_time = st.time_input("Heure d'envoi", key="schedule_time")
            
            st.markdown("---")
            
            # Bouton d'envoi
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                send_now = st.button("📤 Envoyer maintenant", type="primary", use_container_width=True)
            
            with col2:
                if schedule_alert:
                    save_schedule = st.button("💾 Enregistrer la programmation", use_container_width=True)
                else:
                    save_schedule = False
            
            # Traitement de l'envoi
            if send_now:
                success_count = 0
                error_messages = []
                
                if alert_type in ["Email", "Les deux"]:
                    if from_email and email_password and recipient_email:
                        smtp_config = {
                            'smtp_server': smtp_server,
                            'smtp_port': int(smtp_port),
                            'from_email': from_email,
                            'password': email_password
                        }
                        
                        # Afficher un spinner pendant l'envoi
                        with st.spinner("Envoi de l'email en cours..."):
                            success, message = send_email_alert(recipient_email, alert_title, message_body, smtp_config)
                        
                        if success:
                            success_count += 1
                            st.success(f"✅ Email envoyé à {recipient_email}")
                        else:
                            error_messages.append(f"Email : {message}")
                            st.error(f"❌ Erreur email : {message}")
                    else:
                        st.warning("⚠️ Configuration email incomplète")
                
                if alert_type in ["Microsoft Teams", "Les deux"]:
                    if teams_webhook:
                        # Déterminer si c'est un workflow HTTP
                        is_workflow = st.session_state.get('teams_type', 'Webhook') == 'Workflow HTTP'
                        
                        # Construire un message Teams riche avec tous les détails
                        if is_workflow:
                            # Pour les workflows HTTP, format JSON simple
                            teams_message = f"""
                            {alert_title}
                            
                            Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}
                            Nombre d'étudiants concernés : {len(selected_students)}
                            
                            Résumé :
                            - Profils critiques (≥70%) : {len(selected_students[selected_students['risque_score'] >= 0.7])}
                            - Risque élevé (50-70%) : {len(selected_students[(selected_students['risque_score'] >= 0.5) & (selected_students['risque_score'] < 0.7)])}
                            - Risque modéré (30-50%) : {len(selected_students[(selected_students['risque_score'] >= 0.3) & (selected_students['risque_score'] < 0.5)])}
                            
                            Action recommandée : Contacter les étudiants concernés et organiser des entretiens pédagogiques.
                            """
                        else:
                            # Pour les webhooks Teams, construire directement les données structurées
                            # pour créer un MessageCard bien formaté
                            teams_message = {
                                "date": datetime.now().strftime('%d/%m/%Y %H:%M'),
                                "nombre_etudiants": len(selected_students),
                                "resume": {
                                    "critiques": len(selected_students[selected_students['risque_score'] >= 0.7]),
                                    "eleve": len(selected_students[(selected_students['risque_score'] >= 0.5) & (selected_students['risque_score'] < 0.7)]),
                                    "modere": len(selected_students[(selected_students['risque_score'] >= 0.3) & (selected_students['risque_score'] < 0.5)])
                                },
                                "etudiants": []
                            }
                            
                            # Ajouter les étudiants (limiter à 20 pour éviter un message trop long)
                            for idx, row in selected_students.head(20).iterrows():
                                risk_level, emoji, color, _ = get_risk_level(row['risque_score'])
                                teams_message["etudiants"].append({
                                    "id": row['id_etudiant'],
                                    "programme": row['programme'],
                                    "annee": int(row['annee_etude']),
                                    "score": row['risque_score']*100,
                                    "note": row['note_moyenne'],
                                    "emoji": emoji
                                })
                            
                            if len(selected_students) > 20:
                                teams_message["etudiants_restants"] = len(selected_students) - 20
                        
                        # Afficher un spinner pendant l'envoi
                        with st.spinner(f"Envoi de l'alerte Teams ({'Workflow HTTP' if is_workflow else 'Webhook'}) en cours..."):
                            # Si c'est un dict (webhook), passer directement les données structurées
                            if isinstance(teams_message, dict):
                                success, message = send_teams_webhook(teams_webhook, alert_title, teams_message, is_workflow=is_workflow, message_data=teams_message)
                            else:
                                success, message = send_teams_webhook(teams_webhook, alert_title, teams_message, is_workflow=is_workflow)
                        
                        if success:
                            success_count += 1
                            st.success("✅ Alerte Teams envoyée")
                        else:
                            error_messages.append(f"Teams : {message}")
                            st.error(f"❌ Erreur Teams : {message}")
                    else:
                        st.warning("⚠️ URL webhook Teams non configurée")
                
                if success_count > 0:
                    st.balloons()
                    st.success(f"🎉 {success_count} alerte(s) envoyée(s) avec succès !")
                
                if error_messages:
                    for error in error_messages:
                        st.error(error)
            
            if save_schedule and schedule_alert:
                # Sauvegarder la configuration (dans un fichier JSON ou base de données)
                schedule_config = {
                    'frequency': frequency,
                    'time': str(schedule_time),
                    'alert_type': alert_type,
                    'students': selected_students['id_etudiant'].tolist(),
                    'title': alert_title,
                    'message': message_body
                }
                
                try:
                    with open('alert_schedule.json', 'w', encoding='utf-8') as f:
                        json.dump(schedule_config, f, indent=2, ensure_ascii=False)
                    st.success("✅ Programmation enregistrée !")
                    st.info("💡 Note : Pour exécuter les alertes programmées, vous devrez configurer un script cron ou un planificateur de tâches.")
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'enregistrement : {str(e)}")
        else:
            st.warning("Aucun étudiant à risque sélectionné.")
    else:
        st.info("ℹ️ Aucun étudiant à risque dans les filtres sélectionnés.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <small>Tableau de bord POC - EPITECH Bordeaux | Conforme RGPD | Outil d'aide à la décision</small>
</div>
""", unsafe_allow_html=True)
