# Projet de Prédiction du Risque de Décrochage Étudiant - EPITECH Bordeaux

Ce projet vise à prédire le risque de décrochage des étudiants à **EPITECH Bordeaux** et à créer un tableau de bord interactif pour les accompagnateurs et enseignants.

## 🔒 Conformité RGPD et Éthique

**Important** : Ce dataset est conçu pour être **100% conforme au RGPD** et aux normes éthiques :
- ✅ Aucune donnée personnelle identifiable (pas de noms, emails, adresses, etc.)
- ✅ Uniquement des données accessibles par l'établissement
- ✅ IDs anonymisés
- ✅ Données observables et collectables légalement

## Structure du projet

- `generate_student_dataset.py` : Script pour générer un dataset artificiel réaliste EPITECH Bordeaux
- `requirements.txt` : Dépendances Python nécessaires
- `dataset_epitech_bordeaux_decrochage.csv` : Dataset généré (5000 étudiants)

## Installation

```bash
pip install -r requirements.txt
```

## Génération du dataset

```bash
python generate_student_dataset.py
```

Cela générera un fichier `dataset_epitech_bordeaux_decrochage.csv` avec 5000 étudiants et leurs caractéristiques.

## Variables du dataset

Le dataset contient **30 variables** conformes RGPD :

### Informations d'identification (anonymisées)
- `id_etudiant` : Identifiant anonymisé (format: EPI-BDX-XXXXX)
- `etablissement` : EPITECH Bordeaux
- `programme` : Programme suivi (Programme Grande École, Bachelor, MSc)
- `classe` : Classe/Groupe de l'étudiant (ex: PGE-1A-G1, BACH-2A-G3)
- `annee_etude` : Année d'étude (1 à 5)
- `tranche_age` : Tranche d'âge (18-20, 21-23, 24-26, 27+)

### Données académiques
- `note_moyenne` : Note moyenne sur 20
- `note_programmation` : Note en programmation
- `note_projet` : Note moyenne des projets (crucial à EPITECH)
- `note_innovation` : Note en innovation
- `taux_absences` : Taux d'absences (%)
- `nb_absences` : Nombre d'absences
- `nb_retards` : Nombre de retards
- `nb_echec_evaluation` : Nombre d'évaluations échouées

### Données d'engagement (spécifiques EPITECH)
- `participation_projets` : Niveau de participation aux projets (0-1)
- `participation_cours` : Niveau de participation en cours (0-1)
- `participation_activites` : Participation aux activités (0-1)
- `nb_projets_termines` : Nombre de projets terminés
- `nb_projets_en_retard` : Nombre de projets en retard
- `nb_activites_participees` : Nombre d'activités auxquelles l'étudiant a participé

### Données socio-économiques (limitées)
- `boursier` : Statut boursier (0/1) - uniquement oui/non

### Données comportementales observables
- `temps_etude_semaine` : Temps d'étude par semaine (heures)
- `nb_heures_cours_semaine` : Nombre d'heures de cours/workshops par semaine
- `taille_classe` : Taille de la classe (Petite, Moyenne, Grande)

### Interactions avec l'établissement
- `nb_rdv_pedagogique` : Nombre de rendez-vous avec les accompagnateurs
- `nb_demandes_aide` : Nombre de demandes d'aide
- `nb_rappel_discipline` : Nombre de rappels à l'ordre
- `satisfaction_formation` : Satisfaction vis-à-vis de la formation (0-1) - collectée via enquêtes anonymes

### Variables cibles (à prédire par ML)
- `decrochage` : Décrochage (0/1) - **sera prédite par le modèle ML**
- `risque_score` : Score de risque calculé (0-1) - **sera prédite par le modèle ML**

> **Note** : Ces variables ne sont pas incluses dans le dataset généré car elles seront prédites par le modèle de machine learning.

## Caractéristiques du dataset

- **300 étudiants** d'EPITECH Bordeaux
- **3 programmes** : Programme Grande École, Bachelor, MSc
- **28 variables** : Données académiques, d'engagement et comportementales
- **Classes réalistes** : Groupes par programme et année
- **Variables cibles** : `risque_score` et `decrochage` seront prédites par le modèle ML

## Prochaines étapes

1. ✅ Génération du dataset (conforme RGPD)
2. ✅ Création du modèle ML (notebook Jupyter)
3. ✅ Création du tableau de bord interactif (POC)

## Modèle ML

Le notebook `modele_prediction_decrochage.ipynb` contient :
- Chargement et exploration des données
- Création de la variable cible (décrochage et score de risque)
- Préparation des données (encodage, normalisation)
- Entraînement de plusieurs modèles (Random Forest, Gradient Boosting, Logistic Regression)
- Sélection du meilleur modèle
- Visualisation des résultats
- Sauvegarde des modèles pour le déploiement

### Utilisation

```bash
jupyter notebook modele_prediction_decrochage.ipynb
```

Les modèles sauvegardés seront utilisés dans le tableau de bord interactif.

## Tableau de bord interactif (Streamlit)

Le fichier `dashboard_decrochage.py` contient un prototype de tableau de bord interactif (POC) pour les accompagnateurs et enseignants.

### Fonctionnalités

- **🚨 Alertes visuelles** : Mise en avant des profils critiques nécessitant une action immédiate
- **📊 KPIs en temps réel** : Nombre d'étudiants à risque, profils critiques, risque moyen
- **🔍 Filtres interactifs** : Par programme, année d'étude, niveau de risque
- **📈 Visualisations** :
  - Distribution du score de risque
  - Risque par programme et par année
  - Corrélation note/risque avec taille selon absences
  - Graphique radar pour le profil individuel
- **👥 Liste des étudiants à risque** : 
  - Mise en avant visuelle des profils critiques (rouge) et à risque élevé (orange)
  - Tableau triable avec les étudiants prioritaires
- **🔍 Détails par étudiant** : 
  - Vue détaillée avec toutes les caractéristiques
  - Recommandations d'action personnalisées
  - Graphique radar du profil

### Utilisation

1. **Générer le dataset** (si pas déjà fait) :
```bash
python generate_student_dataset.py
```

2. **Entraîner les modèles** (si pas déjà fait) :
   - Ouvrir `modele_prediction_decrochage.ipynb` dans Jupyter
   - Exécuter toutes les cellules pour générer les modèles

3. **Lancer le tableau de bord** :
```bash
streamlit run dashboard_decrochage.py
```

4. **Accéder au tableau de bord** :
   - Le tableau de bord s'ouvrira automatiquement dans votre navigateur
   - URL par défaut : http://localhost:8501

### Intérêt pour les accompagnateurs et enseignants

Le tableau de bord permet de :
- ✅ **Identifier rapidement** les étudiants à risque de décrochage avec alertes visuelles
- ✅ **Prioriser les actions** selon le niveau de risque (critique, élevé, modéré, faible)
- ✅ **Comprendre les facteurs** qui influencent le décrochage via les visualisations
- ✅ **Obtenir des recommandations** personnalisées par étudiant
- ✅ **Suivre l'évolution** par programme et année d'étude
- ✅ **Prendre des décisions éclairées** basées sur les données prédictives
- ✅ **Visualiser le profil** de chaque étudiant avec un graphique radar

