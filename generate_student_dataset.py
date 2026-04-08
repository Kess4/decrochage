"""
Script pour générer un dataset artificiel réaliste sur le décrochage étudiant
à EPITECH Bordeaux, conforme RGPD et normes éthiques.

Ce script génère uniquement des données accessibles par l'établissement,
sans données personnelles identifiables (pas de noms, emails, adresses, etc.).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Configuration de la graine pour la reproductibilité
np.random.seed(42)
random.seed(42)

def generate_epitech_bordeaux_dataset(n_students=300):
    """
    Génère un dataset artificiel réaliste d'étudiants EPITECH Bordeaux 
    avec risque de décrochage, conforme RGPD.
    
    Seules les données accessibles par l'établissement sont incluses :
    - Données académiques (notes, absences, participation)
    - Données d'engagement (projets, présence)
    - Données administratives non-identifiantes (programme, année, classe)
    - Statut boursier (oui/non uniquement)
    
    Parameters:
    -----------
    n_students : int
        Nombre d'étudiants à générer
    
    Returns:
    --------
    pd.DataFrame
        Dataset avec toutes les variables pertinentes et conformes RGPD
    """
    
    data = []
    
    # Programmes EPITECH Bordeaux
    programmes = ['Programme Grande École', 'Bachelor', 'MSc']
    
    # Classes/Groups pour chaque programme
    # Programme Grande École : années 1 à 5
    classes_pge = [f'PGE-{annee}A-G{i}' for annee in range(1, 6) for i in range(1, 9)]  # 8 groupes par année
    
    # Bachelor : années 1 à 3
    classes_bachelor = [f'BACH-{annee}A-G{i}' for annee in range(1, 4) for i in range(1, 6)]  # 5 groupes par année
    
    # MSc : années 1 à 2
    classes_msc = [f'MSc-{annee}A-G{i}' for annee in range(1, 3) for i in range(1, 4)]  # 3 groupes par année
    
    # Technologies/Modules enseignés à EPITECH (exemples réalistes)
    technologies = ['C', 'Python', 'JavaScript', 'Java', 'C++', 'Go', 'Rust', 'Web', 'Mobile', 'DevOps']
    
    for i in range(n_students):
        # Sélection du programme
        programme = np.random.choice(programmes, p=[0.50, 0.35, 0.15])  # PGE majoritaire
        
        # Sélection de la classe selon le programme
        if programme == 'Programme Grande École':
            classe = np.random.choice(classes_pge)
            annee_etude = int(classe.split('-')[1][0])  # Extraire l'année de la classe
        elif programme == 'Bachelor':
            classe = np.random.choice(classes_bachelor)
            annee_etude = int(classe.split('-')[1][0])
        else:  # MSc
            classe = np.random.choice(classes_msc)
            annee_etude = int(classe.split('-')[1][0])
        
        # Facteur de risque latent (corrèle l'ensemble des variables entre elles)
        # Valeur entre 0 et 1, suivant une distribution Beta(2, 5)
        facteur_risque_latent = np.random.beta(2, 5)
        
        # Données démographiques minimales (conformes RGPD)
        # Seulement tranche d'âge, pas d'âge exact
        tranche_age = np.random.choice(['18-20', '21-23', '24-26', '27+'], 
                                       p=[0.40, 0.35, 0.20, 0.05])
        
        # Données académiques EPITECH
        # Notes sur 20 (système français)
        # EPITECH a un système de notation spécifique, mais on utilise 0-20 pour simplicité
        base_note = np.random.normal(12, 3)
        if programme == 'MSc':
            base_note += 1.0  # MSc généralement meilleurs résultats
        if annee_etude >= 3:
            base_note += 0.5  # Meilleures notes en années supérieures
        
        # Le facteur de risque latent fait baisser la note moyenne
        note_moyenne = base_note - 4 * facteur_risque_latent + np.random.normal(0, 0.5)
        note_moyenne = max(5, min(20, round(note_moyenne, 1)))
        
        # Notes par module/technologie (spécifique EPITECH)
        note_programmation = max(5, min(20, round(note_moyenne + np.random.normal(0, 1.0), 1)))
        note_projet = max(5, min(20, round(note_moyenne + np.random.normal(0, 1.5), 1)))
        note_innovation = max(5, min(20, round(note_moyenne + np.random.normal(0, 1.0), 1)))
        
        # Assiduité et absences (données observables par l'établissement)
        # Plus le facteur de risque est élevé, plus le taux d'absences est important
        taux_absences = 5 + 30 * facteur_risque_latent + np.random.normal(0, 2)
        taux_absences = max(0, min(30, taux_absences))  # On borne entre 0% et 30%
        nb_absences = int(np.random.poisson(max(0.1, taux_absences / 8)))
        nb_retards = int(np.random.poisson(max(0.1, taux_absences / 12)))
        
        # Participation aux projets (spécifique EPITECH - pédagogie par projet)
        # Plus le risque est élevé, plus la participation décroît
        participation_projets = 0.9 - 0.7 * facteur_risque_latent + np.random.normal(0, 0.05)
        participation_projets = max(0, min(1, participation_projets))
        nb_projets_termines = int(np.random.poisson(max(0.1, participation_projets * 6)))
        nb_projets_en_retard = int(np.random.poisson(0.5 + 3 * facteur_risque_latent))
        
        # Participation en cours/workshops
        participation_cours = 0.9 - 0.6 * facteur_risque_latent + np.random.normal(0, 0.05)
        participation_cours = max(0, min(1, participation_cours))
        
        # Engagement dans les activités (observable)
        participation_activites = 0.8 - 0.5 * facteur_risque_latent + np.random.normal(0, 0.1)
        participation_activites = max(0, min(1, participation_activites))
        nb_activites_participees = int(np.random.poisson(max(0.1, participation_activites * 4)))
        
        # Données socio-économiques (limitées - conformes RGPD)
        boursier = np.random.choice([0, 1], p=[0.60, 0.40])  # EPITECH a plus de boursiers
        
        # Données comportementales observables
        temps_etude_semaine = np.random.normal(30, 10)  # EPITECH demande beaucoup de travail
        # Les étudiants à risque ont tendance à moins travailler de manière autonome
        temps_etude_semaine = temps_etude_semaine - 6 * facteur_risque_latent
        temps_etude_semaine = max(10, min(60, int(temps_etude_semaine)))
        
        # Heures de cours/workshops par semaine (spécifique EPITECH)
        nb_heures_cours_semaine = np.random.choice([20, 25, 30, 35], p=[0.20, 0.40, 0.30, 0.10])
        
        # Taille de classe (observable)
        taille_classe = np.random.choice(['Petite (<25)', 'Moyenne (25-35)', 'Grande (>35)'],
                                         p=[0.30, 0.50, 0.20])
        
        # Interactions avec l'établissement (observables)
        nb_rdv_pedagogique = int(np.random.poisson(2))  # Nombre de RDV avec accompagnateurs
        nb_demandes_aide = int(np.random.poisson(1.5))
        
        # Satisfaction formation (peut être collectée via enquêtes anonymes)
        satisfaction_formation = 0.8 - 0.6 * facteur_risque_latent + np.random.normal(0, 0.1)
        satisfaction_formation = max(0, min(1, satisfaction_formation))
        
        # Difficultés académiques observables
        nb_rappel_discipline = int(np.random.poisson(0.2 + 1.5 * facteur_risque_latent))  # Rappels à l'ordre
        nb_echec_evaluation = int(np.random.poisson(0.3 + 3 * facteur_risque_latent))  # Évaluations échouées
        
        # Création de l'enregistrement (uniquement données RGPD-compliantes)
        record = {
            'id_etudiant': f'EPI-BDX-{i+1:05d}',  # ID anonymisé
            'etablissement': 'EPITECH Bordeaux',
            'programme': programme,
            'classe': classe,
            'annee_etude': annee_etude,
            'tranche_age': tranche_age,
            'note_moyenne': note_moyenne,
            'note_programmation': note_programmation,
            'note_projet': note_projet,
            'note_innovation': note_innovation,
            'taux_absences': round(taux_absences, 1),
            'nb_absences': nb_absences,
            'nb_retards': nb_retards,
            'participation_projets': round(participation_projets, 2),
            'participation_cours': round(participation_cours, 2),
            'participation_activites': round(participation_activites, 2),
            'nb_projets_termines': nb_projets_termines,
            'nb_projets_en_retard': nb_projets_en_retard,
            'nb_activites_participees': nb_activites_participees,
            'boursier': boursier,
            'temps_etude_semaine': temps_etude_semaine,
            'nb_heures_cours_semaine': nb_heures_cours_semaine,
            'taille_classe': taille_classe,
            'nb_rdv_pedagogique': nb_rdv_pedagogique,
            'nb_demandes_aide': nb_demandes_aide,
            'nb_rappel_discipline': nb_rappel_discipline,
            'nb_echec_evaluation': nb_echec_evaluation,
            'satisfaction_formation': round(satisfaction_formation, 2)
        }
        
        data.append(record)
    
    df = pd.DataFrame(data)
    
    # --- Construction des variables cibles à partir des données observées ---
    # Normalisation min-max des variables utilisées dans le score de risque
    variables_risque = [
        'taux_absences',
        'nb_echec_evaluation',
        'nb_rappel_discipline',
        'nb_projets_en_retard',
        'note_moyenne',
        'participation_projets',
        'participation_cours',
        'satisfaction_formation'
    ]
    
    df_norm = df.copy()
    for col in variables_risque:
        col_min = df[col].min()
        col_max = df[col].max()
        # Évite la division par zéro si la variable est constante
        if col_max > col_min:
            df_norm[col] = (df[col] - col_min) / (col_max - col_min)
        else:
            df_norm[col] = 0.0
    
    # Combinaison pondérée des facteurs de risque et protecteurs
    # Facteurs de risque (poids positifs)
    score_risque = (
        0.25 * df_norm['taux_absences'] +
        0.15 * df_norm['nb_echec_evaluation'] +
        0.10 * df_norm['nb_rappel_discipline'] +
        0.10 * df_norm['nb_projets_en_retard']
    )
    
    # Facteurs protecteurs (poids négatifs)
    score_risque -= (
        0.20 * df_norm['note_moyenne'] +
        0.10 * df_norm['participation_projets'] +
        0.05 * df_norm['participation_cours'] +
        0.05 * df_norm['satisfaction_formation']
    )
    
    # Application d'une sigmoïde pour obtenir un score entre 0 et 1
    risque_score = 1 / (1 + np.exp(-4 * score_risque))
    
    # Ajout d'un bruit gaussien pour le réalisme
    bruit = np.random.normal(0, 0.05, size=len(df))
    risque_score = risque_score + bruit
    risque_score = np.clip(risque_score, 0, 1)
    
    df['risque_score'] = risque_score
    
    # Variable binaire de décrochage (~20% de la population)
    seuil_decrochage = np.quantile(df['risque_score'], 0.80)
    df['decrochage'] = (df['risque_score'] >= seuil_decrochage).astype(int)
    
    return df

if __name__ == "__main__":
    print("=" * 70)
    print("Génération du dataset EPITECH Bordeaux - Conforme RGPD")
    print("=" * 70)
    print("\nCe dataset contient uniquement des données accessibles par l'établissement")
    print("et ne contient aucune donnée personnelle identifiable.\n")
    
    df = generate_epitech_bordeaux_dataset(n_students=300)
    
    # Sauvegarde
    output_file = 'dataset_epitech_bordeaux_decrochage.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n✓ Dataset généré avec succès : {output_file}")
    print(f"✓ Nombre d'étudiants : {len(df)}")
    print(f"✓ Variables : {len(df.columns)} (risque_score et decrochage seront prédits par ML)")
    
    print(f"\n{'='*70}")
    print("Répartition par programme :")
    print(f"{'='*70}")
    print(df['programme'].value_counts())
    
    print(f"\n{'='*70}")
    print("Répartition par année d'étude :")
    print(f"{'='*70}")
    print(df['annee_etude'].value_counts().sort_index())
    
    print(f"\n{'='*70}")
    print("Aperçu des données (5 premières lignes) :")
    print(f"{'='*70}")
    print(df.head())
    
    print(f"\n{'='*70}")
    print("Statistiques descriptives :")
    print(f"{'='*70}")
    print(df.describe())
    
    print(f"\n{'='*70}")
    print("Informations sur le dataset :")
    print(f"{'='*70}")
    print(df.info())
    
    print(f"\n{'='*70}")
    print("✓ Dataset conforme RGPD - Aucune donnée personnelle identifiable")
    print(f"{'='*70}")

