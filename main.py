from RAG import *
from topic_modeling import *



def main():
    """Fonction principale pour exécuter tout le processus."""
    # Ouvrir une fenêtre de dialogue pour sélectionner les fichiers PDF
    root = tk.Tk()
    root.withdraw()  # Masquer la fenêtre principale
    pdf_files = filedialog.askopenfilenames(title="Sélectionnez des fichiers PDF", filetypes=[("PDF files", "*.pdf")])
    if not pdf_files:
        print("Aucun fichier sélectionné. Sortie...")
        sys.exit()
    
    if len(pdf_files) < 2:
        print("Veuillez sélectionner au moins deux fichiers PDF pour la comparaison.")
        return
    
    # Extraction et analyse des topics avec visualisation
    topics1, topics2, similarity = compare_top_topics(pdf_files[0], pdf_files[1])
    
    if topics1 and topics2:
        save_results_to_file(topics1, topics2, similarity, 'results.txt')
    
    message_template = f"""<|system|> Please don't go out of context, respond according to the information you gather from the PDF documents if you don't know, say I don't know, there's no point in talking nonsense.
    Please provide a clear and well-structured answer in the form of an article, i.e. introduction, state of the art, conclusion.
    </s>
    <|user|>
    Question: {QUERY}
    Helpful Answer:
    </s>"""
    # Processus de RAG
    llm = setup_llm()
    embed_model = setup_embed_model()
    documents = read_pdfs(pdf_files)
    setup_service_context(llm)
    storage_context = setup_storage_context()
    index = construct_knowledge_graph_index(documents, storage_context, embed_model)
    query_engine = create_query_engine(index)
    response = generate_response(query_engine,  message_template)
    save_response(response)
    visualize_knowledge_graph(index)

if __name__ == "__main__":
    main()
