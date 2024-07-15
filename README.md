# PDF Comparative Analysis with Topic Modeling and RAG

## Description

Ce projet permet de comparer deux documents PDF en extrayant et en analysant les sujets principaux (topics) de chaque document. Ensuite, il utilise le RAG (Retrieval-Augmented Generation) pour générer des réponses enrichies basées sur ces sujets. Le projet utilise diverses bibliothèques pour le traitement de texte, le machine learning, et la visualisation des résultats.

## Structure du projet

Le projet est organisé en trois fichiers principaux :

1. `main.py` : Point d'entrée du programme.
2. `topic_modeling.py` : Contient les fonctions pour l'extraction et l'analyse des sujets, y compris l'extraction et le prétraitement des textes PDF.
3. `rag.py` : Contient les fonctions pour configurer l'API d'inférence HuggingFace, le modèle d'embeddings, la gestion des requêtes et la visualisation.

### Détails des fichiers

#### `main.py`
- **Description** : Ce fichier gère le flux principal du programme, y compris la sélection des fichiers PDF, l'extraction et l'analyse des topics, la génération de requêtes enrichies, et l'affichage des résultats.
- **Fonction principale** : `main()`

#### `topic_modeling.py`
- **Description** : Ce fichier contient des fonctions pour extraire le texte des fichiers PDF, prétraiter le texte, extraire les topics à l'aide de NMF, et comparer les topics entre deux documents.
- **Fonctions principales** :
  - `extract_text_from_pdf(pdf_path)`
  - `preprocess_text(text)`
  - `extract_and_analyze(pdf_path)`
  - `compare_top_topics(pdf_path1, pdf_path2)`

#### `RAG.py`
- **Description** : Ce fichier contient des fonctions pour configurer l'API d'inférence HuggingFace, le modèle d'embeddings, lire les fichiers PDF, configurer le contexte de service et de stockage, construire l'index du graphe de connaissances, créer un moteur de requêtes, générer des réponses et visualiser les résultats.
- **Fonctions principales** :
  - `setup_llm()`
  - `setup_embed_model()`
  - `read_pdfs(pdf_files)`
  - `setup_service_context(llm)`
  - `setup_storage_context()`
  - `construct_knowledge_graph_index(documents, storage_context, embed_model)`
  - `create_query_engine(index)`
  - `generate_response(query_engine, query)`
  - `save_response(response, filename="response.md")`
  - `visualize_knowledge_graph(index, output_html="Knowledge_graph.html")`

## Prérequis

- Python 3.7+
- Les bibliothèques Python suivantes :
  - os
  - re
  - logging
  - sys
  - matplotlib
  - seaborn
  - numpy
  - PyPDF2
  - scikit-learn
  - nltk
  - tkinter
  - llama_index
  - langchain_community
  - pyvis
  - IPython

## Installation

1. Clonez le dépôt :
   ```bash

   git clone https://github.com/votre-utilisateur/votre-repo.git

   cd votre-repo

3. Installez les dépendances :

    ```bash
    pip install -r requirements.txt

Assurez-vous d'avoir un token API HuggingFace valide et mettez-le à jour dans le fichier RAG.py

## Utilisation
1. Exécutez le script principal :

    ```bash
    python main.py
Sélectionnez au moins deux fichiers PDF pour la comparaison.

Les résultats, y compris les visualisations et les réponses générées, seront affichés et enregistrés dans le répertoire du projet.

Auteur

 - BERKANI Yacine

