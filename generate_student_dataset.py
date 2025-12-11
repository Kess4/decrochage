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
        
        note_moyenne = max(5, min(20, round(base_note, 1)))
        
        # Notes par module/technologie (spécifique EPITECH)
        note_programmation = max(5, min(20, round(base_note + np.random.normal(0, 1.5), 1)))
        note_projet = max(5, min(20, round(base_note + np.random.normal(0, 2), 1)))
        note_innovation = max(5, min(20, round(base_note + np.random.normal(0, 1.5), 1)))
        
        # Assiduité et absences (données observables par l'établissement)
        taux_absences = np.random.beta(2, 5) * 25  # Entre 0 et 25%
        nb_absences = int(np.random.poisson(taux_absences / 8))
        nb_retards = int(np.random.poisson(taux_absences / 12))
        
        # Participation aux projets (spécifique EPITECH - pédagogie par projet)
        participation_projets = np.random.beta(3, 2)  # Entre 0 et 1
        nb_projets_termines = int(np.random.poisson(participation_projets * 5))
        nb_projets_en_retard = int(np.random.poisson((1 - participation_projets) * 2))
        
        # Participation en cours/workshops
        participation_cours = np.random.beta(3, 2)  # Entre 0 et 1
        
        # Engagement dans les activités (observable)
        participation_activites = np.random.beta(2, 3)  # Entre 0 et 1
        nb_activites_participees = int(np.random.poisson(participation_activites * 3))
        
        # Données socio-économiques (limitées - conformes RGPD)
        boursier = np.random.choice([0, 1], p=[0.60, 0.40])  # EPITECH a plus de boursiers
        
        # Données comportementales observables
        temps_etude_semaine = np.random.normal(30, 10)  # EPITECH demande beaucoup de travail
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
        satisfaction_formation = np.random.beta(3, 2)
        
        # Difficultés académiques observables
        nb_rappel_discipline = int(np.random.poisson(0.3))  # Rappels à l'ordre
        nb_echec_evaluation = int(np.random.poisson(0.5))  # Évaluations échouées
        
        # Création de l'enregistrement (uniquement données RGPD-compliantes)
        # Note: risque_score et decrochage seront prédits par le modèle ML
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

