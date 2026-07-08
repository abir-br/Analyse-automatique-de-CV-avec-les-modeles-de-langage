# Analyse et Classement Automatique de CV par Similarité Sémantique

Application web permettant de classer automatiquement des CV (PDF) par pertinence par rapport à une offre d'emploi, à l'aide d'**embeddings de phrases** (`sentence-transformers`) et de la **similarité cosinus**. Développée avec **Flask**, **PyMuPDF** et **scikit-learn**.

Projet réalisé dans le cadre d'un TER (Travail d'Étude et de Recherche) — voir `RAPPORT PROJET TER.pdf` pour le détail méthodologique, et `RecruitAI_Code_GBL_ALM/Documentation.pdf` pour la documentation technique du code.

---

## Structure du projet

```
Analyse-automatique-de-CV-avec-les-modeles-de-langage-main/
├── RAPPORT PROJET TER.pdf             
└── RecruitAI_Code_GBL_ALM/
    ├── app.py                          
    ├── requirements.txt                
    ├── Documentation.pdf                
    ├── cv_to_texte.ipynb                
    ├── cv_to_texte.pdf                  
    ├── Strategy_choice.ipynb            
    ├── Strategy_choice.pdf             
    ├── static/
    │   └── style.css                  
    └── templates/
        └── index.html                  

---

## Description du fonctionnement

### 1. Extraction du texte des CV (`extraire_texte_cv`)
Le texte est extrait des PDF via **PyMuPDF (fitz)**, avec un traitement avancé pour gérer les CV mal structurés :
- **Détection de mise en colonnes** (`clusteriser_colonnes`, `est_vraie_mise_en_colonnes`) : regroupe les lignes par position horizontale et détecte si le CV est en plusieurs colonnes (fréquent sur les CV modernes).
- **Reconstruction de l'ordre de lecture** (`ordonner_clusters`, `reconstruire_lignes_ordonnee`) : lit la colonne principale puis les colonnes secondaires, dans le bon ordre.
- **Nettoyage du texte** (`nettoyer_cv_final`, `supprimer_emojis_et_symboles_inutiles`) : suppression des emojis/symboles, normalisation des puces, conversion des barres de progression graphiques (`███░░`) en niveaux textuels (`niveau 3/5`).
- **OCR de secours** (`pytesseract`) : activable pour les CV scannés (image) si le texte extrait est trop court.

>  Toute cette logique a été mise au point et testée pas à pas dans le notebook `cv_to_texte.ipynb`, avant son intégration dans `app.py`.

### 2. Prétraitement et découpage en blocs (chunking)
Avant l'analyse sémantique, chaque CV est découpé en blocs (`pretraiter_cv` + stratégie de chunking). Quatre stratégies principales sont proposées dans l'interface, plus une cinquième utilisée en interne :

| Stratégie      | Fonction              | Principe                                                                 |
|----------------|------------------------|---------------------------------------------------------------------------|
| **Adaptatif** *(recommandée)* | `chunk_adaptatif`   | Découpage hiérarchique (paragraphes → lignes → phrases → mots) puis fusion sémantique des blocs similaires selon un seuil adaptatif |
| **Sauts de ligne** | `chunk_by_newlines` | Découpage simple par double saut de ligne                                |
| **Taille fixe**    | `chunk_by_words`    | Blocs de taille fixe (80 mots) avec chevauchement (20 mots)               |
| **Par phrases**    | `chunk_by_sentences`| Regroupement de phrases jusqu'à une limite de mots (100)                  |
| Sémantique *(interne)* | `chunk_semantic` | Découpe aux points de rupture où la similarité entre phrases consécutives chute |

### 3. Classement des CV (`classer_cvs`)
1. L'offre d'emploi est encodée en embedding avec le modèle multilingue `paraphrase-multilingual-MiniLM-L12-v2`.
2. Chaque CV est découpé en blocs selon la stratégie choisie, puis chaque bloc est encodé.
3. La **similarité cosinus** entre l'offre et chaque bloc du CV est calculée.
4. Le meilleur score (bloc le plus pertinent) est retenu comme score global du CV.
5. Les CV sont triés par score décroissant.

### 4. Interface web (`templates/index.html`, `static/style.css`)
- Zone de saisie/import de l'offre d'emploi (texte, PDF ou TXT).
- Zone de **drag & drop** pour l'upload de plusieurs CV en PDF.
- Sélecteur de stratégie de chunking.
- Affichage des résultats classés avec score de similarité et extrait du bloc le plus pertinent.


## Notebooks d'expérimentation

Le projet inclut deux notebooks Jupyter documentant la phase de recherche/mise au point, en amont de l'application Flask :

### `cv_to_texte.ipynb`
Développement et test pas à pas des fonctions d'extraction et de nettoyage de texte à partir des PDF (nettoyage des emojis, gestion des colonnes, conversion des barres de compétences...), avant leur intégration dans `app.py`.

### `Strategy_choice.ipynb`
Comparaison quantitative des différentes stratégies de chunking sur un jeu de 20 CV de test, face à une **offre d'emploi** et un **classement de référence** attendu (rang 1 à 20). Pour chaque stratégie (sauts de ligne, taille fixe, phrases, sémantique, adaptatif, et une **stratégie de fusion pondérée** combinant les scores de toutes les stratégies), le notebook :
- calcule le classement obtenu par similarité cosinus,
- évalue sa qualité par rapport au classement attendu à l'aide de métriques de classement (corrélation de **Spearman**, **NDCG**).

Ce notebook constitue la justification expérimentale du choix de la stratégie **adaptative** comme stratégie recommandée par défaut dans l'application.

---

## Installation

```bash
cd RecruitAI_Code_GBL_ALM

# (recommandé) créer un environnement virtuel
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```



## Exécution

```bash
python app.py
```

Le serveur démarre sur `http://localhost:5000` (chargement du modèle `sentence-transformers` au démarrage, ce qui peut prendre quelques secondes).

---

