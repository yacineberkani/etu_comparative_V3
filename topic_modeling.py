from __init__ import *



nltk.download('stopwords')

# Fonction pour extraire le texte d'un PDF
def extract_text_from_pdf(pdf_path):
    """Extrait le texte d'un fichier PDF."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Fonction pour prétraiter le texte
def preprocess_text(text):
    text = re.sub(r'https?:\/\/\S+', '', text)
    text = re.sub(r'\b\d{4}\b', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()

    stop_words = set(stopwords.words('english')).union(ENGLISH_STOP_WORDS)
    stop_words = list(stop_words)

    def is_valid_token(token):
        return len(token) > 2 and token not in stop_words

    tokens = [word for word in text.split() if is_valid_token(word)]
    cleaned_text = ' '.join(tokens)

    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=500, ngram_range=(1, 2))
    vectorizer.fit_transform([cleaned_text])

    return cleaned_text, vectorizer

# Fonction pour extraire et analyser les topics
def extract_and_analyze(pdf_path):
    extracted_text = extract_text_from_pdf(pdf_path)
    preprocessed_text, vectorizer = preprocess_text(extracted_text)
    
    num_topics = 5
    nmf = NMF(n_components=num_topics, random_state=42).fit(vectorizer.transform([preprocessed_text]))
    
    topics = []
    topic_weights = []
    for topic_idx, topic in enumerate(nmf.components_):
        topic_words = [vectorizer.get_feature_names_out()[i] for i in sorted(range(len(topic)), key=lambda i: topic[i])[-5:]]
        topics.extend(topic_words)
        weights = sorted(topic)[-5:]
        topic_weights.extend(weights)

    topics = [x for _, x in sorted(zip(topic_weights, topics), reverse=True)]
    topic_weights = sorted(topic_weights, reverse=True)
    
    return topics[:7], topic_weights[:7]

def cosine_similarity_keywords(topics_pdf1, topics_pdf2):
    """
    Calcule la similarité cosinus entre deux listes de mots-clés.
    
    Args:
    topics_pdf1 (list of str): Liste de mots-clés pour le premier document.
    topics_pdf2 (list of str): Liste de mots-clés pour le deuxième document.
    
    Returns:
    float: Valeur de la similarité cosinus entre les deux listes de mots-clés.
    """
    # Créer un vecteur de comptage
    vectorizer = CountVectorizer().fit_transform([' '.join(topics_pdf1), ' '.join(topics_pdf2)])
    vectors = vectorizer.toarray()

    # Calculer la similarité cosinus
    cosine_sim = cosine_similarity(vectors)
    
    # Retourner la similarité cosinus entre les deux vecteurs
    return cosine_sim[0, 1]



def compare_top_topics(pdf_path1, pdf_path2):
    topics1, weights1 = extract_and_analyze(pdf_path1)
    topics2, weights2 = extract_and_analyze(pdf_path2)

    if not topics1 or not topics2:
        print("Erreur dans l'extraction des topics, impossible de comparer les documents.")
        return None, None, None

    print("Topics for PDF 1:", topics1)
    print("Topics for PDF 2:", topics2)
    
    similarity = cosine_similarity_keywords(topics1, topics2)
    

    fig, axs = plt.subplots(1, 2, figsize=(14, 8))

    sns.barplot(x=weights1, y=topics1, ax=axs[0], palette="viridis")
    axs[0].set_title('Topics for PDF 1')
    axs[0].set_xlabel('Poids')
    axs[0].set_ylabel('Topics')

    sns.barplot(x=weights2, y=topics2, ax=axs[1], palette="viridis")
    axs[1].set_title('Topics for PDF 2')
    axs[1].set_xlabel('Poids')
    axs[1].set_ylabel('Topics')

    plt.tight_layout()
    plt.savefig('img.png')
    plt.close()

    return topics1, topics2, similarity

def save_results_to_file(topics1, topics2, similarity, file_path):
    with open(file_path, 'w') as file:
        file.write("Topics for PDF 1:\n")
        file.write(", ".join(topics1) + "\n\n")
        file.write("Topics for PDF 2:\n")
        file.write(", ".join(topics2) + "\n\n")
        file.write(f"Similarity between topics: {similarity}\n")
