"""
Tableau de bord — Prédiction du Risque de Décrochage
EPITECH Bordeaux | MVP / POC à destination des équipes pédagogiques

Cible utilisateur : accompagnateurs pédagogiques, responsables de programme,
intervenants — profils non-techniques.

Données sources :
  - predictions_test.csv  (généré par le notebook de modélisation)
  - model_decrochage.pkl  + preprocessor.pkl  (optionnels, pour nouvelles prédictions)

Conformité RGPD : aucune donnée identifiante, données synthétiques uniquement.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import warnings
import smtplib
import json
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EPITECH Bordeaux — Suivi décrochage",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "critique":  "#E53935",
    "eleve":     "#F57C00",
    "modere":    "#F9A825",
    "faible":    "#43A047",
    "primary":   "#1A237E",
    "secondary": "#283593",
    "surface":   "#F5F7FA",
    "border":    "#E0E4EC",
    "text":      "#1C2340",
    "muted":     "#6B7280",
}

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {{
        font-family: 'DM Sans', sans-serif;
        color: {COLORS['text']};
    }}

    /* ── Fond général ── */
    .stApp {{ background: {COLORS['surface']}; }}

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {{
        background: {COLORS['primary']};
    }}
    section[data-testid="stSidebar"] * {{
        color: white !important;
    }}
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stMultiSelect label {{
        color: rgba(255,255,255,0.75) !important;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }}
    section[data-testid="stSidebar"] .stSelectbox > div > div {{
        background: rgba(255,255,255,0.12) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        color: white !important;
        border-radius: 8px;
    }}

    /* ── Onglets ── */
    .stTabs [data-baseweb="tab-list"] {{
        background: white;
        border-radius: 12px;
        padding: 4px;
        border: 1px solid {COLORS['border']};
        gap: 2px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        font-weight: 500;
        font-size: 14px;
        padding: 8px 20px;
        color: {COLORS['muted']};
    }}
    .stTabs [aria-selected="true"] {{
        background: {COLORS['primary']} !important;
        color: white !important;
    }}

    /* ── Métriques ── */
    [data-testid="metric-container"] {{
        background: white;
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }}
    [data-testid="metric-container"] [data-testid="stMetricLabel"] {{
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: {COLORS['muted']};
    }}
    [data-testid="metric-container"] [data-testid="stMetricValue"] {{
        font-size: 28px;
        font-weight: 700;
        color: {COLORS['text']};
    }}

    /* ── Cartes ── */
    .card {{
        background: white;
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 12px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }}
    .card-critique {{ border-left: 4px solid {COLORS['critique']}; }}
    .card-eleve    {{ border-left: 4px solid {COLORS['eleve']}; }}
    .card-modere   {{ border-left: 4px solid {COLORS['modere']}; }}
    .card-faible   {{ border-left: 4px solid {COLORS['faible']}; }}

    /* ── Badges ── */
    .badge {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }}
    .badge-critique {{ background: #FEECEC; color: {COLORS['critique']}; }}
    .badge-eleve    {{ background: #FEF3E8; color: {COLORS['eleve']}; }}
    .badge-modere   {{ background: #FFFBEA; color: #B45309; }}
    .badge-faible   {{ background: #ECFDF5; color: {COLORS['faible']}; }}

    /* ── Bannière alerte ── */
    .alert-banner {{
        background: linear-gradient(135deg, {COLORS['critique']} 0%, #B71C1C 100%);
        color: white;
        padding: 14px 20px;
        border-radius: 10px;
        margin-bottom: 8px;
        font-weight: 600;
        font-size: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    .alert-banner-warn {{
        background: linear-gradient(135deg, {COLORS['eleve']} 0%, #E65100 100%);
        color: white;
        padding: 12px 20px;
        border-radius: 10px;
        margin-bottom: 8px;
        font-weight: 500;
        font-size: 14px;
    }}

    /* ── Bandeau indicateur individuel ── */
    .student-header {{
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 20px;
    }}

    /* ── Facteur tag ── */
    .factor-tag {{
        display: inline-block;
        background: #EEF2FF;
        color: {COLORS['secondary']};
        border: 1px solid #C7D2FE;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 12px;
        font-family: 'DM Mono', monospace;
        margin: 2px;
    }}

    /* ── Note outil POC ── */
    .poc-note {{
        background: #EEF2FF;
        border: 1px solid #C7D2FE;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 12px;
        color: {COLORS['secondary']};
        margin-top: 6px;
    }}

    /* ── Masquer éléments Streamlit superflus ── */
    #MainMenu, footer {{ visibility: hidden; }}
    .block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; }}
    h1 {{ font-size: 22px !important; font-weight: 700 !important; }}
    h2 {{ font-size: 18px !important; font-weight: 600 !important; }}
    h3 {{ font-size: 15px !important; font-weight: 600 !important; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def hex_to_rgba(hex_color: str, alpha: float = 0.18) -> str:
    """Convertit un hex (#RRGGBB) en chaîne rgba() compatible Plotly."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def get_risk_level(score: float) -> tuple:
    """Retourne (label, emoji, css_class, couleur_hex) selon le score de risque."""
    if score >= 0.70:
        return "Critique", "🔴", "critique", COLORS["critique"]
    elif score >= 0.50:
        return "Élevé",    "🟠", "eleve",    COLORS["eleve"]
    elif score >= 0.30:
        return "Modéré",   "🟡", "modere",   COLORS["modere"]
    else:
        return "Faible",   "🟢", "faible",   COLORS["faible"]


def score_bar_html(score: float, color: str) -> str:
    """Mini barre de progression HTML inline."""
    pct = int(score * 100)
    return (
        f"<div style='background:#E0E4EC;border-radius:4px;height:8px;width:100%;'>"
        f"<div style='background:{color};width:{pct}%;height:100%;border-radius:4px;'></div>"
        f"</div><small style='color:{color};font-weight:700;'>{pct}%</small>"
    )


def get_actions(risk_label: str, row: pd.Series) -> list[str]:
    """Actions pédagogiques contextuelles selon le niveau de risque et le profil."""
    actions = []

    if risk_label == "Critique":
        actions.append("🚨 **Entretien individuel urgent** à planifier sous 48h avec le responsable pédagogique")
        actions.append("📧 **Notifier le responsable de programme** pour un suivi coordonné")
    elif risk_label == "Élevé":
        actions.append("📅 **Planifier un RDV pédagogique** dans les 2 semaines")
        actions.append("🎯 **Proposer un tutorat ciblé** sur les modules en difficulté")
    elif risk_label == "Modéré":
        actions.append("✉️ **Envoyer un email de suivi bienveillant** avec ressources d'aide")
        actions.append("👀 **Surveiller l'évolution** lors du prochain bilan mensuel")
    else:
        actions.append("✅ **Profil stable** — maintenir le suivi régulier de cohorte")

    # Actions complémentaires selon les signaux individuels
    if row.get("taux_absences", 0) > 20:
        actions.append("📞 **Contacter l'étudiant** pour comprendre les absences répétées (>20%)")
    if row.get("note_moyenne", 20) < 9:
        actions.append("📚 **Soutien académique** : accompagnement renforcé sur les matières fondamentales")
    if row.get("nb_projets_en_retard", 0) >= 2:
        actions.append("⏱️ **Atelier gestion du temps** : proposer un coaching en organisation de projets")
    if row.get("satisfaction_formation", 1) < 0.35:
        actions.append("💬 **Entretien motivation** : explorer les causes d'insatisfaction vis-à-vis de la formation")
    if row.get("nb_rdv_pedagogique", 99) == 0:
        actions.append("🤝 **Aucun RDV à ce jour** — initier le premier contact pédagogique")

    return actions


# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONS D'ENVOI D'ALERTES
# ─────────────────────────────────────────────────────────────────────────────
def send_email_alert(to_email, subject, body, smtp_config=None):
    """Envoie un email d'alerte HTML."""
    if smtp_config is None:
        return False, "Configuration SMTP non définie"
    try:
        msg = MIMEMultipart()
        msg["From"]    = smtp_config["from_email"]
        msg["To"]      = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "html"))
        port = int(smtp_config["smtp_port"])
        if port == 465:
            server = smtplib.SMTP_SSL(smtp_config["smtp_server"], port)
        else:
            server = smtplib.SMTP(smtp_config["smtp_server"], port)
            server.starttls()
        server.login(smtp_config["from_email"], smtp_config["password"])
        server.send_message(msg)
        server.quit()
        return True, "Email envoyé avec succès"
    except smtplib.SMTPAuthenticationError:
        return False, "Erreur d'authentification : vérifiez email et mot de passe"
    except smtplib.SMTPConnectError as e:
        return False, f"Erreur de connexion SMTP : {e}"
    except (smtplib.SMTPServerDisconnected, ConnectionError) as e:
        return False, f"Connexion interrompue : {e}"
    except Exception as e:
        return False, f"Erreur : {e}"


def send_teams_webhook(webhook_url, title, message, color="E53935", is_workflow=False, message_data=None):
    """Envoie une alerte Teams via webhook (MessageCard) ou Workflow HTTP."""
    try:
        if is_workflow:
            if isinstance(message, dict):
                msg_text = f"{title}\n\nDate : {message.get('date','')}\n"
                msg_text += f"Étudiants concernés : {message.get('nombre_etudiants',0)}\n\n"
                resume = message.get("resume", {})
                msg_text += f"🔴 Critiques (≥70%) : {resume.get('critiques',0)}\n"
                msg_text += f"🟠 Élevé (50-70%) : {resume.get('eleve',0)}\n"
                msg_text += f"🟡 Modéré (30-50%) : {resume.get('modere',0)}\n\nListe :\n"
                for e in message.get("etudiants", []):
                    msg_text += f"- {e['id']} ({e['programme']}, A{e['annee']}) : {e['emoji']} {e['score']:.1f}% | Note {e['note']:.1f}/20\n"
                if message.get("etudiants_restants", 0) > 0:
                    msg_text += f"... et {message['etudiants_restants']} autre(s)\n"
                msg_text += "\nAction : contacter les étudiants et organiser des entretiens pédagogiques."
                message = msg_text
            payload = {"title": title, "message": message, "color": color}
        else:
            # MessageCard enrichie
            if isinstance(message, dict):
                resume = message.get("resume", {})
                sections = [
                    {
                        "activityTitle": title,
                        "activitySubtitle": f"Date : {message.get('date', '')}",
                        "facts": [{"name": "Étudiants concernés :", "value": str(message.get("nombre_etudiants", 0))}],
                    },
                    {
                        "title": "Résumé :",
                        "facts": [
                            {"name": "🔴 Critiques (≥70%)", "value": str(resume.get("critiques", 0))},
                            {"name": "🟠 Élevé (50-70%)",   "value": str(resume.get("eleve",    0))},
                            {"name": "🟡 Modéré (30-50%)",  "value": str(resume.get("modere",   0))},
                        ],
                    },
                ]
                etudiants_facts = []
                for e in message.get("etudiants", [])[:15]:
                    etudiants_facts.append({
                        "name":  f"• {e['id']}",
                        "value": f"{e['programme']} — Année {e['annee']} | {e['emoji']} {e['score']:.1f}% | Note {e['note']:.1f}/20",
                    })
                if message.get("etudiants_restants", 0) > 0:
                    etudiants_facts.append({"name": "...", "value": f"{message['etudiants_restants']} autre(s)"})
                sections.append({"title": "Liste des étudiants :", "facts": etudiants_facts})
                sections.append({"title": "Action recommandée :", "text": "Contacter les étudiants et organiser des entretiens pédagogiques."})
            else:
                sections = [{"activityTitle": title, "activitySubtitle": datetime.now().strftime("%d/%m/%Y %H:%M"), "text": str(message)}]
            payload = {
                "@type":    "MessageCard",
                "@context": "http://schema.org/extensions",
                "themeColor": color,
                "summary":    title,
                "sections":   sections,
            }
        response = requests.post(webhook_url, json=payload, timeout=10)
        if response.status_code in [200, 201, 202, 204]:
            return True, "Alerte Teams envoyée avec succès"
        return False, f"Erreur HTTP {response.status_code} : {response.text[:200]}"
    except requests.exceptions.Timeout:
        return False, "Timeout : la requête a pris trop de temps"
    except requests.exceptions.ConnectionError:
        return False, "Erreur de connexion : vérifiez votre accès réseau"
    except Exception as e:
        return False, f"Erreur : {e}"



@st.cache_data(show_spinner="Chargement des données…")
def load_data() -> tuple[pd.DataFrame | None, str | None]:
    try:
        df = pd.read_csv("predictions_test.csv", encoding="utf-8-sig")

        # ── 1. Supprimer les colonnes dupliquées (garde la première occurrence) ──
        df = df.loc[:, ~df.columns.duplicated()]

        # ── 2. Réinitialiser l'index pour éviter les doublons d'index ──
        df = df.reset_index(drop=True)

        # ── 3. Harmonisation des noms de colonnes cibles ──
        rename = {}
        if "risque_score_predit" in df.columns and "risque_score" not in df.columns:
            rename["risque_score_predit"] = "risque_score"
        if "decrochage_predit" in df.columns and "decrochage_pred" not in df.columns:
            rename["decrochage_predit"] = "decrochage_pred"
        if rename:
            df = df.rename(columns=rename)

        # ── 4. Fallback risque_score original si colonne prédite absente ──
        if "risque_score" not in df.columns and "risque_score_orig" in df.columns:
            df["risque_score"] = df["risque_score_orig"]

        # ── 5. Garantir que decrochage_pred existe (fallback sur seuil 0.4) ──
        if "decrochage_pred" not in df.columns and "risque_score" in df.columns:
            df["decrochage_pred"] = (df["risque_score"] >= 0.40).astype(int)

        return df, None
    except FileNotFoundError:
        return None, (
            "Le fichier **predictions_test.csv** est introuvable. "
            "Veuillez d'abord exécuter le notebook `modele_prediction_decrochage.ipynb`."
        )


@st.cache_resource(show_spinner=False)
def load_models() -> tuple:
    try:
        model = joblib.load("model_decrochage.pkl")
        prep  = joblib.load("preprocessor.pkl")
        return (model, prep), None
    except FileNotFoundError:
        return None, "Modèles non trouvés — les prédictions pré-calculées seront utilisées."


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT + GARDE
# ─────────────────────────────────────────────────────────────────────────────
df_raw, err = load_data()
if df_raw is None:
    st.error(f"❌ {err}")
    st.stop()

models_result, models_warn = load_models()
if models_warn:
    # Avertissement discret, pas bloquant
    pass

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — FILTRES
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='padding:16px 0 8px 0;'>"
        "<span style='font-size:22px;'>🎓</span>"
        "<span style='font-size:15px;font-weight:700;margin-left:8px;'>EPITECH Bordeaux</span>"
        "<br><span style='font-size:11px;opacity:0.6;'>Outil de suivi décrochage — POC</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border-color:rgba(255,255,255,0.15);margin:8px 0 16px 0;'>", unsafe_allow_html=True)

    st.markdown("<p style='font-size:11px;font-weight:700;opacity:0.6;letter-spacing:0.08em;margin-bottom:4px;'>FILTRES</p>", unsafe_allow_html=True)

    programmes = ["Tous"] + sorted(df_raw["programme"].dropna().unique().tolist())
    sel_prog = st.selectbox("Programme", programmes)

    annees = ["Toutes"] + sorted(df_raw["annee_etude"].dropna().unique().tolist())
    sel_annee = st.selectbox("Année d'étude", annees)

    sel_risque = st.selectbox(
        "Niveau de risque",
        ["Tous", "🔴 Critique (≥70%)", "🟠 Élevé (50–70%)", "🟡 Modéré (30–50%)", "🟢 Faible (<30%)"],
    )

    st.markdown("<hr style='border-color:rgba(255,255,255,0.15);margin:16px 0 12px 0;'>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:11px;font-weight:700;opacity:0.6;letter-spacing:0.08em;margin-bottom:4px;'>DONNÉES</p>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='font-size:12px;opacity:0.8;'>{len(df_raw)} étudiants chargés<br>"
        f"Mise à jour : {datetime.now().strftime('%d/%m/%Y')}</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='poc-note' style='background:rgba(255,255,255,0.1);border-color:rgba(255,255,255,0.2);color:rgba(255,255,255,0.7);margin-top:8px;'>"
        "⚠️ Outil d'aide à la décision — les prédictions ne remplacent pas le jugement pédagogique."
        "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION DES FILTRES
# ─────────────────────────────────────────────────────────────────────────────
df = df_raw.copy()

if sel_prog != "Tous":
    df = df[df["programme"] == sel_prog]
if sel_annee != "Toutes":
    df = df[df["annee_etude"] == sel_annee]
if "Critique" in sel_risque:
    df = df[df["risque_score"] >= 0.70]
elif "Élevé" in sel_risque:
    df = df[(df["risque_score"] >= 0.50) & (df["risque_score"] < 0.70)]
elif "Modéré" in sel_risque:
    df = df[(df["risque_score"] >= 0.30) & (df["risque_score"] < 0.50)]
elif "Faible" in sel_risque:
    df = df[df["risque_score"] < 0.30]

n_total    = len(df)
n_critique = len(df[df["risque_score"] >= 0.70])
n_eleve    = len(df[(df["risque_score"] >= 0.50) & (df["risque_score"] < 0.70)])
n_modere   = len(df[(df["risque_score"] >= 0.30) & (df["risque_score"] < 0.50)])
n_faible   = len(df[df["risque_score"] < 0.30])
tx_decrochage = df["decrochage_pred"].mean() * 100 if n_total > 0 else 0.0

# ─────────────────────────────────────────────────────────────────────────────
# EN-TÊTE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"<h1 style='margin-bottom:2px;'>Prédiction du risque de décrochage</h1>"
    f"<p style='color:{COLORS['muted']};font-size:13px;margin-top:0;'>EPITECH Bordeaux — Outil d'aide à la décision pédagogique</p>",
    unsafe_allow_html=True,
)

# Alertes critiques visibles dès l'entrée
if n_critique > 0:
    st.markdown(
        f"<div class='alert-banner'>🚨 {n_critique} étudiant{'s' if n_critique > 1 else ''} "
        f"nécessite{'nt' if n_critique > 1 else ''} une action immédiate (risque critique ≥ 70%)</div>",
        unsafe_allow_html=True,
    )
if n_eleve > 0:
    st.markdown(
        f"<div class='alert-banner-warn'>⚠️ {n_eleve} étudiant{'s' if n_eleve > 1 else ''} "
        f"présente{'nt' if n_eleve > 1 else ''} un risque élevé (50–70%) — suivi renforcé recommandé</div>",
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ONGLETS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Vue d'ensemble",
    "👥  Étudiants à risque",
    "🔍  Profil individuel",
    "📈  Analyses",
    "🔔  Alertes",
])


# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 1 — Vue d'ensemble
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Étudiants suivis", n_total)
    with c2:
        st.metric("🔴 Critique", n_critique,
                  delta=f"{n_critique/n_total*100:.1f}%" if n_total else "—",
                  delta_color="inverse")
    with c3:
        st.metric("🟠 Élevé", n_eleve,
                  delta=f"{n_eleve/n_total*100:.1f}%" if n_total else "—",
                  delta_color="inverse")
    with c4:
        st.metric("🟡 Modéré", n_modere)
    with c5:
        st.metric("Taux décrochage prédit", f"{tx_decrochage:.1f}%",
                  delta_color="inverse")

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # Graphiques macro
    col_left, col_right = st.columns([1, 1])

    with col_left:
        fig_donut = go.Figure(go.Pie(
            labels=["Critique (≥70%)", "Élevé (50–70%)", "Modéré (30–50%)", "Faible (<30%)"],
            values=[n_critique, n_eleve, n_modere, n_faible],
            hole=0.55,
            marker_colors=[COLORS["critique"], COLORS["eleve"], COLORS["modere"], COLORS["faible"]],
            textinfo="percent+label",
            hovertemplate="%{label}<br>%{value} étudiants<extra></extra>",
        ))
        fig_donut.add_annotation(
            text=f"<b>{n_total}</b><br><span style='font-size:11px'>étudiants</span>",
            x=0.5, y=0.5, showarrow=False, font_size=16,
        )
        fig_donut.update_layout(
            title="Répartition par niveau de risque",
            showlegend=False,
            margin=dict(t=40, b=10, l=10, r=10),
            height=300,
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_right:
        risque_prog = (
            df.groupby("programme")["risque_score"]
            .agg(risque_moyen="mean", nb="count")
            .reset_index()
        )
        risque_prog["risque_moyen_pct"] = risque_prog["risque_moyen"] * 100
        risque_prog["label"] = risque_prog["nb"].astype(str) + " étudiants"

        fig_prog = px.bar(
            risque_prog,
            x="programme",
            y="risque_moyen_pct",
            color="risque_moyen_pct",
            color_continuous_scale=[[0, COLORS["faible"]], [0.5, COLORS["modere"]], [1, COLORS["critique"]]],
            range_color=[0, 100],
            text="label",
            labels={"programme": "Programme", "risque_moyen_pct": "Risque moyen (%)"},
            title="Risque moyen par programme",
        )
        fig_prog.update_traces(textposition="outside")
        fig_prog.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            yaxis=dict(range=[0, 100], title="Risque moyen (%)"),
            margin=dict(t=40, b=20, l=20, r=20),
            height=300,
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        st.plotly_chart(fig_prog, use_container_width=True)

    # Évolution par année
    risque_annee = (
        df.groupby("annee_etude")["risque_score"]
        .agg(risque_moyen="mean", nb="count")
        .reset_index()
    )
    risque_annee["risque_moyen_pct"] = risque_annee["risque_moyen"] * 100

    fig_annee = px.bar(
        risque_annee,
        x="annee_etude",
        y="risque_moyen_pct",
        color="risque_moyen_pct",
        color_continuous_scale=[[0, COLORS["faible"]], [0.5, COLORS["modere"]], [1, COLORS["critique"]]],
        range_color=[0, 100],
        text="nb",
        labels={"annee_etude": "Année d'étude", "risque_moyen_pct": "Risque moyen (%)"},
        title="Risque moyen par année d'étude",
    )
    fig_annee.update_traces(texttemplate="%{text} étudiants", textposition="outside")
    fig_annee.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        yaxis=dict(range=[0, 100]),
        xaxis=dict(dtick=1),
        margin=dict(t=40, b=20, l=20, r=20),
        height=280,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    st.plotly_chart(fig_annee, use_container_width=True)

    # Note POC
    st.markdown(
        "<div class='poc-note'>"
        "📌 <strong>Note méthodologique :</strong> Les scores de risque sont des probabilités prédites "
        "par un modèle de machine learning entraîné sur des données synthétiques réalistes (EPITECH Bordeaux). "
        "Ce prototype vise à identifier précocement les étudiants nécessitant un accompagnement — "
        "il constitue un <em>outil d'aide à la décision</em>, pas un outil de décision automatique."
        "</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 2 — Étudiants à risque
# ══════════════════════════════════════════════════════════════════════════════
with tab2:

    df_risque = df[df["risque_score"] >= 0.30].sort_values("risque_score", ascending=False).copy()

    if df_risque.empty:
        st.info("✅ Aucun étudiant à risque dans les filtres sélectionnés.")
    else:
        # Sous-filtres rapides
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("🔴 Critiques", len(df_risque[df_risque["risque_score"] >= 0.70]))
        with c2:
            st.metric("🟠 Élevés",   len(df_risque[(df_risque["risque_score"] >= 0.50) & (df_risque["risque_score"] < 0.70)]))
        with c3:
            st.metric("🟡 Modérés",  len(df_risque[(df_risque["risque_score"] >= 0.30) & (df_risque["risque_score"] < 0.50)]))

        st.markdown("---")
        st.subheader("Liste des étudiants nécessitant un suivi")
        st.caption("Triée par score de risque décroissant · Seuls les profils Modéré / Élevé / Critique sont affichés")

        # Colonnes disponibles avec facteurs SHAP si présents
        has_factors = all(c in df_risque.columns for c in ["top_facteur_1", "top_facteur_2", "top_facteur_3"])

        display_cols = {
            "id_etudiant":         "ID",
            "programme":           "Programme",
            "annee_etude":         "Année",
            "note_moyenne":        "Note moy. (/20)",
            "taux_absences":       "Absences (%)",
            "participation_projets": "Participation projets",
            "nb_projets_en_retard":  "Projets en retard",
            "risque_score":        "Score de risque",
            "decrochage_pred":     "Décrochage prédit",
        }
        if has_factors:
            display_cols.update({
                "top_facteur_1": "1er facteur de risque",
                "top_facteur_2": "2e facteur de risque",
                "top_facteur_3": "3e facteur de risque",
            })

        df_show = df_risque[[c for c in display_cols.keys() if c in df_risque.columns]].copy()
        df_show.rename(columns=display_cols, inplace=True)

        # Formatage
        if "Score de risque" in df_show.columns:
            df_show["Score de risque"] = df_show["Score de risque"].apply(lambda x: f"{x*100:.1f}%")
        if "Participation projets" in df_show.columns:
            df_show["Participation projets"] = df_show["Participation projets"].apply(lambda x: f"{x*100:.0f}%")
        if "Décrochage prédit" in df_show.columns:
            df_show["Décrochage prédit"] = df_show["Décrochage prédit"].apply(lambda x: "⚠️ Oui" if x == 1 else "Non")
        if "Note moy. (/20)" in df_show.columns:
            df_show["Note moy. (/20)"] = df_show["Note moy. (/20)"].apply(lambda x: f"{x:.1f}")
        if "Absences (%)" in df_show.columns:
            df_show["Absences (%)"] = df_show["Absences (%)"].apply(lambda x: f"{x:.1f}%")

        # Niveau de risque textuel
        df_show.insert(1, "Niveau", df_risque["risque_score"].apply(
            lambda x: "🔴 Critique" if x >= 0.70 else ("🟠 Élevé" if x >= 0.50 else "🟡 Modéré")
        ))

        st.dataframe(df_show, use_container_width=True, height=500, hide_index=True)

        # Légende
        st.markdown("""
        <div style='background:white;border:1px solid #E0E4EC;border-radius:8px;padding:10px 16px;font-size:12px;margin-top:8px;'>
        <strong>Légende :</strong>
        &nbsp;&nbsp;🔴 <strong>Critique</strong> ≥ 70% — action immédiate
        &nbsp;|&nbsp; 🟠 <strong>Élevé</strong> 50–70% — suivi renforcé
        &nbsp;|&nbsp; 🟡 <strong>Modéré</strong> 30–50% — surveillance continue
        <br><small style='color:#6B7280;'>Les facteurs de risque proviennent de l'analyse SHAP (contributions individuelles du modèle ML).</small>
        </div>
        """, unsafe_allow_html=True)

        # Export CSV
        csv_export = df_show.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "⬇️  Exporter la liste (CSV)",
            data=csv_export,
            file_name=f"etudiants_a_risque_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=False,
        )


# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 3 — Profil individuel
# ══════════════════════════════════════════════════════════════════════════════
with tab3:

    if n_total == 0:
        st.warning("Aucun étudiant ne correspond aux filtres sélectionnés.")
    else:
        # Sélecteur — trié par score de risque décroissant pour accès rapide aux urgences
        df_sorted_sel = df.sort_values("risque_score", ascending=False)
        options = df_sorted_sel["id_etudiant"].tolist()

        selected_id = st.selectbox(
            "Sélectionner un étudiant (trié par score de risque décroissant)",
            options,
            key="profil_selector",
        )

        row = df[df["id_etudiant"] == selected_id].iloc[0]
        risk_label, risk_emoji, risk_css, risk_color = get_risk_level(row["risque_score"])
        score_pct = row["risque_score"] * 100

        # ── Bandeau identité + score ──
        bg_light = hex_to_rgba(risk_color, 0.09)
        border_light = hex_to_rgba(risk_color, 0.20)
        st.markdown(
            f"<div class='student-header' style='background:{bg_light};border:1px solid {border_light};'>"
            f"<div style='display:flex;align-items:center;justify-content:space-between;'>"
            f"<div>"
            f"<span style='font-size:20px;font-weight:700;'>{risk_emoji} {selected_id}</span>"
            f"&nbsp;&nbsp;<span class='badge badge-{risk_css}'>{risk_label}</span>"
            f"<p style='margin:4px 0 0 0;font-size:13px;color:{COLORS['muted']};'>"
            f"{row.get('programme','—')} · Année {int(row.get('annee_etude',0))} · {row.get('classe','—')}"
            f"</p></div>"
            f"<div style='text-align:right;'>"
            f"<span style='font-size:36px;font-weight:800;color:{risk_color};'>{score_pct:.0f}%</span>"
            f"<p style='font-size:11px;color:{COLORS['muted']};margin:0;'>score de risque</p>"
            f"</div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Actions recommandées ──
        st.markdown("### ✅ Actions recommandées")
        actions = get_actions(risk_label, row)
        for a in actions:
            st.markdown(f"- {a}")

        # Facteurs SHAP si disponibles
        has_factors = all(c in row.index for c in ["top_facteur_1", "top_facteur_2", "top_facteur_3"])
        if has_factors:
            st.markdown("**Principaux facteurs de risque détectés par le modèle :**")
            factors_html = "".join([
                f"<span class='factor-tag'>{row[f]}</span>"
                for f in ["top_facteur_1", "top_facteur_2", "top_facteur_3"]
                if pd.notna(row.get(f))
            ])
            st.markdown(factors_html + "<br><small style='color:#6B7280;'>Source : valeurs SHAP (contribution individuelle de chaque variable au score)</small>", unsafe_allow_html=True)

        st.markdown("---")

        # ── Deux colonnes : données académiques + engagement ──
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### 📚 Parcours académique")
            def kpi_row(label, value, warning=False):
                color = COLORS["critique"] if warning else COLORS["text"]
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;padding:6px 0;"
                    f"border-bottom:1px solid {COLORS['border']};'>"
                    f"<span style='font-size:13px;color:{COLORS['muted']};'>{label}</span>"
                    f"<span style='font-size:13px;font-weight:600;color:{color};'>{value}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            kpi_row("Note moyenne", f"{row.get('note_moyenne',0):.1f} / 20", row.get("note_moyenne", 20) < 10)
            kpi_row("Note programmation", f"{row.get('note_programmation',0):.1f} / 20", row.get("note_programmation", 20) < 10)
            kpi_row("Note projet", f"{row.get('note_projet',0):.1f} / 20", row.get("note_projet", 20) < 10)
            kpi_row("Note innovation", f"{row.get('note_innovation',0):.1f} / 20")
            kpi_row("Satisfaction formation", f"{row.get('satisfaction_formation',0)*100:.0f}%", row.get("satisfaction_formation", 1) < 0.35)
            kpi_row("Statut boursier", "Oui" if row.get("boursier", 0) == 1 else "Non")

        with c2:
            st.markdown("#### 📊 Engagement & comportement")
            kpi_row("Taux d'absences", f"{row.get('taux_absences',0):.1f}%", row.get("taux_absences", 0) > 20)
            kpi_row("Nombre d'absences", str(int(row.get("nb_absences", 0))), int(row.get("nb_absences", 0)) > 5)
            kpi_row("Retards", str(int(row.get("nb_retards", 0))))
            kpi_row("Participation projets", f"{row.get('participation_projets',0)*100:.0f}%", row.get("participation_projets", 1) < 0.30)
            kpi_row("Projets terminés / en retard",
                    f"{int(row.get('nb_projets_termines',0))} / {int(row.get('nb_projets_en_retard',0))}",
                    int(row.get("nb_projets_en_retard", 0)) >= 2)
            kpi_row("RDV pédagogiques", str(int(row.get("nb_rdv_pedagogique", 0))), int(row.get("nb_rdv_pedagogique", 99)) == 0)
            kpi_row("Demandes d'aide", str(int(row.get("nb_demandes_aide", 0))))
            kpi_row("Rappels à l'ordre", str(int(row.get("nb_rappel_discipline", 0))), int(row.get("nb_rappel_discipline", 0)) >= 2)
            kpi_row("Échecs évaluations", str(int(row.get("nb_echec_evaluation", 0))), int(row.get("nb_echec_evaluation", 0)) >= 3)

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

        # ── Graphique radar : profil vs cohorte ──
        st.markdown("#### 📡 Profil comparé à la cohorte (même programme × même année)")

        cohorte = df_raw[
            (df_raw["programme"] == row.get("programme"))
            & (df_raw["annee_etude"] == row.get("annee_etude"))
        ]

        cats = ["Notes", "Assiduité", "Participation\nprojets", "Participation\ncours", "Satisfaction"]
        vals_etudiant = [
            row.get("note_moyenne", 0) / 20,
            max(0, 1 - row.get("taux_absences", 0) / 30),
            row.get("participation_projets", 0),
            row.get("participation_cours", 0),
            row.get("satisfaction_formation", 0),
        ]
        vals_median = [
            cohorte["note_moyenne"].median() / 20 if not cohorte.empty else 0.5,
            max(0, 1 - cohorte["taux_absences"].median() / 30) if not cohorte.empty else 0.5,
            cohorte["participation_projets"].median() if not cohorte.empty else 0.5,
            cohorte["participation_cours"].median() if not cohorte.empty else 0.5,
            cohorte["satisfaction_formation"].median() if not cohorte.empty else 0.5,
        ]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_etudiant + [vals_etudiant[0]],
            theta=cats + [cats[0]],
            fill="toself",
            name=selected_id,
            line_color=risk_color,
            fillcolor=hex_to_rgba(risk_color, 0.18),
            line_width=2,
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_median + [vals_median[0]],
            theta=cats + [cats[0]],
            fill="toself",
            name=f"Médiane cohorte ({len(cohorte)} étudiants)",
            line_color="#607D8B",
            fillcolor="rgba(96,125,139,0.15)",
            line_dash="dot",
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickfont_size=10)),
            showlegend=True,
            height=420,
            margin=dict(t=30, b=20, l=40, r=40),
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown(
            "<div class='poc-note'>Les valeurs du radar sont normalisées (0 = risque maximal, 1 = optimal). "
            "La médiane de cohorte est calculée sur les étudiants du même programme et de la même année d'étude.</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 4 — Analyses
# ══════════════════════════════════════════════════════════════════════════════
with tab4:

    st.subheader("Distribution des scores de risque")

    fig_hist = px.histogram(
        df,
        x="risque_score",
        nbins=30,
        color_discrete_sequence=[COLORS["primary"]],
        labels={"risque_score": "Score de risque", "count": "Nb d'étudiants"},
        opacity=0.8,
    )
    fig_hist.add_vline(x=0.70, line_dash="dash", line_color=COLORS["critique"],
                       annotation_text="Critique (70%)", annotation_position="top right")
    fig_hist.add_vline(x=0.50, line_dash="dash", line_color=COLORS["eleve"],
                       annotation_text="Élevé (50%)", annotation_position="top right")
    fig_hist.add_vline(x=0.30, line_dash="dash", line_color=COLORS["modere"],
                       annotation_text="Modéré (30%)", annotation_position="top right")
    fig_hist.update_layout(
        height=280, paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(t=20, b=20, l=20, r=20),
        yaxis_title="Nombre d'étudiants",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Notes vs Score de risque")
        fig_scatter = px.scatter(
            df,
            x="note_moyenne",
            y="risque_score",
            color="risque_score",
            color_continuous_scale=[[0, COLORS["faible"]], [0.5, COLORS["modere"]], [1, COLORS["critique"]]],
            range_color=[0, 1],
            size="taux_absences",
            size_max=18,
            hover_data=["id_etudiant", "programme", "annee_etude"],
            labels={"note_moyenne": "Note moyenne (/20)", "risque_score": "Score de risque"},
            opacity=0.75,
        )
        fig_scatter.update_layout(
            coloraxis_showscale=False,
            height=320, paper_bgcolor="white", plot_bgcolor="white",
            margin=dict(t=10, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("La taille des points représente le taux d'absences.")

    with col_r:
        st.subheader("Absences vs Score de risque")
        fig_abs = px.scatter(
            df,
            x="taux_absences",
            y="risque_score",
            color="risque_score",
            color_continuous_scale=[[0, COLORS["faible"]], [0.5, COLORS["modere"]], [1, COLORS["critique"]]],
            range_color=[0, 1],
            size="note_moyenne",
            size_max=18,
            hover_data=["id_etudiant", "programme"],
            labels={"taux_absences": "Taux d'absences (%)", "risque_score": "Score de risque"},
            opacity=0.75,
        )
        fig_abs.update_layout(
            coloraxis_showscale=False,
            height=320, paper_bgcolor="white", plot_bgcolor="white",
            margin=dict(t=10, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_abs, use_container_width=True)
        st.caption("La taille des points représente la note moyenne.")

    st.markdown("---")
    st.subheader("Corrélations entre indicateurs clés")

    numeric_cols = [c for c in [
        "note_moyenne", "taux_absences", "participation_projets",
        "participation_cours", "nb_projets_en_retard", "nb_echec_evaluation",
        "nb_rappel_discipline", "satisfaction_formation", "risque_score",
    ] if c in df.columns]

    corr = df[numeric_cols].corr().round(2)
    labels_fr = {
        "note_moyenne": "Note moy.",
        "taux_absences": "Absences",
        "participation_projets": "Part. projets",
        "participation_cours": "Part. cours",
        "nb_projets_en_retard": "Projets retard",
        "nb_echec_evaluation": "Échecs éval.",
        "nb_rappel_discipline": "Rappels",
        "satisfaction_formation": "Satisfaction",
        "risque_score": "Score risque",
    }
    corr.index   = [labels_fr.get(c, c) for c in corr.index]
    corr.columns = [labels_fr.get(c, c) for c in corr.columns]

    fig_corr = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu",
        zmin=-1, zmax=1,
        aspect="auto",
    )
    fig_corr.update_layout(
        height=420,
        paper_bgcolor="white",
        margin=dict(t=10, b=10, l=10, r=10),
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    st.caption(
        "Les corrélations élevées avec **Score risque** confirment les variables les plus prédictives. "
        "Les corrélations négatives indiquent des facteurs protecteurs (ex. note moyenne → risque réduit)."
    )

    # Facteurs les plus fréquents (si SHAP disponible)
    if all(c in df.columns for c in ["top_facteur_1", "top_facteur_2", "top_facteur_3"]):
        st.markdown("---")
        st.subheader("Facteurs de risque les plus fréquents (SHAP)")
        st.caption("Parmi les étudiants à risque Modéré / Élevé / Critique uniquement")

        df_risk_only = df[df["risque_score"] >= 0.30]
        all_factors = pd.concat([
            df_risk_only["top_facteur_1"],
            df_risk_only["top_facteur_2"],
            df_risk_only["top_facteur_3"],
        ]).dropna()

        factor_counts = all_factors.value_counts().head(12).reset_index()
        factor_counts.columns = ["Variable", "Occurrences"]

        fig_factors = px.bar(
            factor_counts,
            x="Occurrences",
            y="Variable",
            orientation="h",
            color="Occurrences",
            color_continuous_scale=[[0, "#C7D2FE"], [1, COLORS["primary"]]],
            labels={"Occurrences": "Nb de fois dans le Top 3 SHAP", "Variable": ""},
        )
        fig_factors.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            yaxis=dict(autorange="reversed"),
            height=360,
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(t=10, b=10, l=20, r=20),
        )
        st.plotly_chart(fig_factors, use_container_width=True)
        st.caption(
            "Ce graphique montre les variables qui reviennent le plus souvent parmi les 3 principaux "
            "facteurs de risque SHAP des étudiants à risque. Il indique sur quels leviers l'accompagnement "
            "pédagogique devrait en priorité se concentrer."
        )



# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 5 — Alertes
# ══════════════════════════════════════════════════════════════════════════════
with tab5:

    st.subheader("Envoi d'alertes aux équipes pédagogiques")
    st.markdown(
        "<p style='color:#6B7280;font-size:13px;'>Sélectionnez les étudiants à risque "
        "et envoyez une alerte par email et/ou Microsoft Teams à l'équipe concernée. "
        "Cette fonctionnalité démontre l'intégrabilité du prototype dans les outils "
        "institutionnels existants.</p>",
        unsafe_allow_html=True,
    )

    # ── Type d'alerte ──
    st.markdown("#### ⚙️ Canal d'envoi")
    alert_type = st.radio(
        "Type d'alerte",
        ["Email", "Microsoft Teams", "Les deux"],
        horizontal=True,
        key="alert_type_radio",
    )

    # ── Config Email ──
    if alert_type in ["Email", "Les deux"]:
        st.markdown("##### 📧 Configuration Email")
        c1, c2 = st.columns(2)
        with c1:
            smtp_server   = st.text_input("Serveur SMTP", value="smtp.gmail.com", key="smtp_server")
            smtp_port     = st.number_input("Port SMTP", value=587, key="smtp_port")
        with c2:
            from_email    = st.text_input("Email expéditeur", key="from_email")
            email_password = st.text_input("Mot de passe", type="password", key="email_password")
        recipient_email = st.text_input("Email destinataire", key="recipient_email")

    # ── Config Teams ──
    if alert_type in ["Microsoft Teams", "Les deux"]:
        st.markdown("##### 💬 Configuration Microsoft Teams")
        teams_type = st.radio(
            "Type de connexion",
            ["Webhook", "Workflow HTTP"],
            horizontal=True,
            help="Webhook : Teams > Connexions > Webhooks entrants | Workflow HTTP : Power Automate / Logic Apps",
            key="teams_type",
        )
        teams_webhook = st.text_input(
            f"URL {'du workflow HTTP' if teams_type == 'Workflow HTTP' else 'du webhook'}",
            help="Copiez l'URL depuis Teams (Gestion des canaux > Connecteurs > Webhooks entrants)",
            key="teams_webhook",
        )

    st.markdown("---")

    # ── Sélection des étudiants ──
    st.markdown("#### 📋 Étudiants à inclure dans l'alerte")

    df_alertes = df[df["risque_score"] >= 0.30].sort_values("risque_score", ascending=False).copy()

    if df_alertes.empty:
        st.info("✅ Aucun étudiant à risque dans les filtres actifs.")
    else:
        selection_mode = st.radio(
            "Mode de sélection",
            ["Tous les étudiants à risque", "Critiques uniquement (≥70%)", "Sélection manuelle"],
            horizontal=True,
            key="selection_mode",
        )

        if selection_mode == "Tous les étudiants à risque":
            selected_students = df_alertes.copy()
        elif selection_mode == "Critiques uniquement (≥70%)":
            selected_students = df_alertes[df_alertes["risque_score"] >= 0.70].copy()
        else:
            student_list = df_alertes["id_etudiant"].tolist()
            selected_ids = st.multiselect(
                "Sélectionner les étudiants",
                student_list,
                default=student_list[:5] if len(student_list) > 5 else student_list,
                key="manual_selection",
            )
            selected_students = df_alertes[df_alertes["id_etudiant"].isin(selected_ids)].copy()

        if len(selected_students) > 0:
            st.info(f"📌 **{len(selected_students)} étudiant(s)** sélectionné(s) pour l'alerte")

            with st.expander("👀 Aperçu des étudiants sélectionnés"):
                prev = selected_students[["id_etudiant", "programme", "annee_etude", "risque_score", "note_moyenne"]].copy()
                prev["risque_score"] = prev["risque_score"].apply(lambda x: f"{x*100:.1f}%")
                prev.columns = ["ID Étudiant", "Programme", "Année", "Score Risque", "Note Moyenne"]
                st.dataframe(prev, hide_index=True, use_container_width=True)

            st.markdown("---")

            # ── Personnalisation du message ──
            st.markdown("#### ✍️ Message")
            alert_title = st.text_input(
                "Titre",
                value=f"🚨 Alerte Décrochage — {len(selected_students)} étudiant(s) à risque",
                key="alert_title",
            )

            message_template = st.selectbox(
                "Modèle de message",
                ["Automatique (recommandé)", "Personnalisé"],
                key="message_template",
            )

            if message_template == "Automatique (recommandé)":
                n_crit = len(selected_students[selected_students["risque_score"] >= 0.70])
                n_elev = len(selected_students[(selected_students["risque_score"] >= 0.50) & (selected_students["risque_score"] < 0.70)])
                n_mod  = len(selected_students[(selected_students["risque_score"] >= 0.30) & (selected_students["risque_score"] < 0.50)])

                rows_html = ""
                for _, r in selected_students.iterrows():
                    rl, em, _, rc = get_risk_level(r["risque_score"])
                    rows_html += (
                        f"<tr>"
                        f"<td style='padding:6px 10px;'>{r['id_etudiant']}</td>"
                        f"<td style='padding:6px 10px;'>{r['programme']}</td>"
                        f"<td style='padding:6px 10px;text-align:center;'>{int(r['annee_etude'])}</td>"
                        f"<td style='padding:6px 10px;color:{rc};font-weight:700;'>{em} {r['risque_score']*100:.1f}%</td>"
                        f"<td style='padding:6px 10px;'>{r['note_moyenne']:.1f}/20</td>"
                        f"</tr>"
                    )

                message_body = f"""
<h2 style="color:#1A237E;">🚨 Alerte — Étudiants à risque de décrochage</h2>
<p><strong>Date :</strong> {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
<p><strong>Nombre d'étudiants concernés :</strong> {len(selected_students)}</p>
<h3>Résumé</h3>
<ul>
  <li>🔴 Profils critiques (≥70%) : <strong>{n_crit}</strong></li>
  <li>🟠 Risque élevé (50–70%) : <strong>{n_elev}</strong></li>
  <li>🟡 Risque modéré (30–50%) : <strong>{n_mod}</strong></li>
</ul>
<h3>Liste des étudiants</h3>
<table border="1" cellpadding="0" cellspacing="0" style="border-collapse:collapse;width:100%;font-size:13px;">
  <thead style="background:#1A237E;color:white;">
    <tr>
      <th style="padding:8px 10px;">ID Étudiant</th>
      <th style="padding:8px 10px;">Programme</th>
      <th style="padding:8px 10px;">Année</th>
      <th style="padding:8px 10px;">Score Risque</th>
      <th style="padding:8px 10px;">Note Moyenne</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>
<p style="margin-top:16px;"><strong>Action recommandée :</strong> Contacter les étudiants concernés
et organiser des entretiens pédagogiques dans les meilleurs délais.</p>
<hr style="margin-top:20px;">
<p style="font-size:11px;color:#6B7280;">
  Envoyé depuis le tableau de bord de prédiction du risque de décrochage — EPITECH Bordeaux<br>
  Outil d'aide à la décision | Données synthétiques | Conforme RGPD
</p>
"""
            else:
                message_body = st.text_area(
                    "Message personnalisé (HTML supporté)",
                    height=200,
                    key="custom_message",
                )

            with st.expander("👀 Aperçu du message"):
                st.markdown(message_body, unsafe_allow_html=True)

            st.markdown("---")

            # ── Programmation récurrente ──
            st.markdown("#### ⏰ Programmation (optionnel)")
            schedule_alert = st.checkbox("Programmer une alerte récurrente", key="schedule_alert")
            if schedule_alert:
                c1, c2 = st.columns(2)
                with c1:
                    frequency = st.selectbox("Fréquence", ["Quotidienne", "Hebdomadaire", "Mensuelle"], key="alert_frequency")
                with c2:
                    if frequency == "Hebdomadaire":
                        st.selectbox("Jour", ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi"], key="schedule_day")
                    elif frequency == "Mensuelle":
                        st.number_input("Jour du mois (1–28)", min_value=1, max_value=28, value=1, key="schedule_day_num")
                    st.time_input("Heure d'envoi", key="schedule_time")

            st.markdown("---")

            # ── Boutons d'action ──
            c1, c2, _ = st.columns([1, 1, 2])
            with c1:
                send_now = st.button("📤 Envoyer maintenant", type="primary", use_container_width=True)
            with c2:
                save_schedule = False
                if schedule_alert:
                    save_schedule = st.button("💾 Enregistrer la programmation", use_container_width=True)

            # ── Traitement envoi ──
            if send_now:
                success_count  = 0
                error_messages = []

                if alert_type in ["Email", "Les deux"]:
                    if from_email and email_password and recipient_email:
                        smtp_cfg = {
                            "smtp_server": smtp_server,
                            "smtp_port":   int(smtp_port),
                            "from_email":  from_email,
                            "password":    email_password,
                        }
                        with st.spinner("Envoi de l'email…"):
                            ok, msg = send_email_alert(recipient_email, alert_title, message_body, smtp_cfg)
                        if ok:
                            success_count += 1
                            st.success(f"✅ Email envoyé à {recipient_email}")
                        else:
                            error_messages.append(f"Email : {msg}")
                            st.error(f"❌ {msg}")
                    else:
                        st.warning("⚠️ Configuration email incomplète (serveur, email expéditeur, mot de passe, destinataire)")

                if alert_type in ["Microsoft Teams", "Les deux"]:
                    if teams_webhook:
                        is_workflow = st.session_state.get("teams_type", "Webhook") == "Workflow HTTP"
                        teams_data = {
                            "date": datetime.now().strftime("%d/%m/%Y %H:%M"),
                            "nombre_etudiants": len(selected_students),
                            "resume": {
                                "critiques": len(selected_students[selected_students["risque_score"] >= 0.70]),
                                "eleve":     len(selected_students[(selected_students["risque_score"] >= 0.50) & (selected_students["risque_score"] < 0.70)]),
                                "modere":    len(selected_students[(selected_students["risque_score"] >= 0.30) & (selected_students["risque_score"] < 0.50)]),
                            },
                            "etudiants": [],
                        }
                        for _, r in selected_students.head(20).iterrows():
                            rl, em, _, _ = get_risk_level(r["risque_score"])
                            teams_data["etudiants"].append({
                                "id": r["id_etudiant"], "programme": r["programme"],
                                "annee": int(r["annee_etude"]), "score": r["risque_score"] * 100,
                                "note": r["note_moyenne"], "emoji": em,
                            })
                        if len(selected_students) > 20:
                            teams_data["etudiants_restants"] = len(selected_students) - 20

                        with st.spinner(f"Envoi Teams ({'Workflow HTTP' if is_workflow else 'Webhook'})…"):
                            ok, msg = send_teams_webhook(
                                teams_webhook, alert_title, teams_data,
                                is_workflow=is_workflow, message_data=teams_data,
                            )
                        if ok:
                            success_count += 1
                            st.success("✅ Alerte Teams envoyée")
                        else:
                            error_messages.append(f"Teams : {msg}")
                            st.error(f"❌ {msg}")
                    else:
                        st.warning("⚠️ URL webhook Teams non renseignée")

                if success_count > 0:
                    st.balloons()
                    st.success(f"🎉 {success_count} alerte(s) envoyée(s) avec succès !")

            if save_schedule and schedule_alert:
                schedule_config = {
                    "frequency":  st.session_state.get("alert_frequency", "Hebdomadaire"),
                    "alert_type": alert_type,
                    "students":   selected_students["id_etudiant"].tolist(),
                    "title":      alert_title,
                }
                try:
                    with open("alert_schedule.json", "w", encoding="utf-8") as f:
                        json.dump(schedule_config, f, indent=2, ensure_ascii=False)
                    st.success("✅ Programmation enregistrée (alert_schedule.json)")
                    st.info("💡 Pour exécuter les alertes programmées, configurez un cron ou un planificateur de tâches pointant vers ce script.")
                except Exception as e:
                    st.error(f"❌ Erreur d'enregistrement : {e}")
        else:
            st.warning("Aucun étudiant sélectionné.")



st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
st.markdown(
    f"<div style='text-align:center;font-size:11px;color:{COLORS['muted']};padding:12px 0 20px;border-top:1px solid {COLORS['border']};'>"
    f"EPITECH Bordeaux — Prototype de prédiction du risque de décrochage · Conforme RGPD · Données synthétiques"
    f"<br>Outil d'aide à la décision pédagogique — v1.0 MVP"
    f"</div>",
    unsafe_allow_html=True,
)