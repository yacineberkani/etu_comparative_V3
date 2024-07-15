from __init__ import *


# Configuration de la journalisation
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Définition des constantes
HF_TOKEN = "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"  # Remplacer par votre API HuggingFace
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct" 
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
QUERY = "Please do a comparative study on these PDF documents"


def extract_text(pdf_path, max_pages=5):
    """Extrait le texte d'un fichier PDF."""
    doc = fitz.open(pdf_path)
    text = ""
    num_pages = min(len(doc), max_pages)
    for page_num in range(num_pages):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def clean_text(text):
    return text.strip()

def preprocess_text(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    processed_text = " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
    return processed_text

def split_text(text, max_chunk_size=1024):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        current_length += len(word) + 1
        if current_length > max_chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = len(word) + 1
        current_chunk.append(word)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def summarize_text(chunks):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    return ' '.join(summaries)

def setup_llm():
    return HuggingFaceInferenceAPI(model_name=MODEL_NAME, token=HF_TOKEN)

def setup_embed_model():
    return LangchainEmbedding(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME))

def extract_and_summarize_pdf(pdf_path):
    text = extract_text(pdf_path)
    clean_text_content = clean_text(text)
    processed_text = preprocess_text(clean_text_content)
    chunks = split_text(processed_text)
    summary = summarize_text(chunks)
    return summary

def read_pdfs(pdf_paths):
    documents = []
    for pdf_path in pdf_paths:
        summary = extract_and_summarize_pdf(pdf_path)
        document = Document(text=summary, metadata={"source": pdf_path})
        documents.append(document)
    return documents


def setup_service_context(llm):
    Settings.llm = llm
    Settings.chunk_size = 1200

def setup_storage_context():
    graph_store = SimpleGraphStore()
    return StorageContext.from_defaults(graph_store=graph_store)

def construct_knowledge_graph_index(documents, storage_context, embed_model):
    return KnowledgeGraphIndex.from_documents(
        documents=documents,
        max_triplets_per_chunk=5,
        storage_context=storage_context,
        embed_model=embed_model,
        include_embeddings=True
    )

def create_query_engine(index):
    return index.as_query_engine(
        include_text=True,
        response_mode="tree_summarize",
        embedding_mode="hybrid",
        similarity_top_k=3,
    )

def generate_response(query_engine, query):
    response = query_engine.query(query)
    return response

def save_response(response, filename="response.md"):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(response.response.split("<|assistant|>")[-1].strip())
    print("Le document a été sauvegardé avec succès")

def visualize_knowledge_graph(index, output_html="Knowledge_graph.html"):
    g = index.get_networkx_graph()
    net = Network(notebook=True, cdn_resources="in_line", directed=True)
    net.from_nx(g)
    net.show("graph.html")
    net.save_graph(output_html)
    display(HTML(filename=output_html))
    print("Le graphe de connaissances a été sauvegardé et affiché avec succès.")
