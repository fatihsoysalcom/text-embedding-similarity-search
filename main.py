import numpy as np
# The article discusses "embeddings" and "similarity search" as core concepts
# for vector indexing. SentenceTransformer provides an easy way to generate
# semantic embeddings from text, which are then used for similarity calculations.
from sentence_transformers import SentenceTransformer

# Function to calculate cosine similarity between two vectors
def calculate_cosine_similarity(vec1, vec2):
    """
    Calculates the cosine similarity between two numpy vectors.
    Cosine similarity measures the cosine of the angle between two vectors.
    A value of 1 means identical direction, 0 means orthogonal, -1 means opposite.
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0 # Handle cases where one or both vectors are zero vectors
    
    return dot_product / (norm_vec1 * norm_vec2)

def main():
    # 1. Define a set of sample text documents.
    # These represent the "knowledge base" from which we want to retrieve information.
    documents = [
        "Yapay zeka ve doğal dil işleme uygulamaları.",
        "Büyük veri kümelerinden hızlı ve doğru bilgi çekme.",
        "Vektör indeksleme teknolojileri, Kurumsal RAG sistemlerinde devrim yaratıyor.",
        "HNSW ve IVF-PQ, yüksek performanslı vektör indeksleme yöntemleridir.",
        "Modern yapay zeka uygulamaları, verileri sayısal vektörlere dönüştürür.",
        "Kedi ve köpekler evcil hayvanlardır, genellikle sevimli olurlar.",
        "Araba sürmek, ulaşım için yaygın bir yöntemdir."
    ]

    # 2. Load a pre-trained SentenceTransformer model.
    # This model converts human-readable text into high-dimensional numerical vectors (embeddings).
    # 'all-MiniLM-L6-v2' is a lightweight model suitable for quick demonstrations.
    print("Yapay zeka uygulamaları için metinleri vektörlere dönüştüren modeli yüklüyor...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model başarıyla yüklendi.")

    # 3. Generate embeddings for all documents.
    # Each document is transformed into a vector, representing its semantic content.
    print("\nBelgeler için vektör gömmeleri (embeddings) oluşturuluyor...")
    document_embeddings = model.encode(documents, show_progress_bar=True)
    print(f"Toplam {len(document_embeddings)} belge için {document_embeddings.shape[1]} boyutlu vektörler oluşturuldu.")

    # 4. Define a query text.
    # This is the user's input for which we want to find similar information.
    query_text = "Vektör indeksleme ve yapay zeka uygulamaları"

    # 5. Generate an embedding for the query.
    # The query is also converted into a vector using the same model.
    print(f"\nSorgu metni için vektör gömmesi oluşturuluyor: '{query_text}'")
    query_embedding = model.encode(query_text)

    # 6. Calculate cosine similarity between the query embedding and each document embedding.
    # This step determines how semantically similar the query is to each document.
    print("Sorgu ile belgeler arasındaki benzerlikler hesaplanıyor...")
    similarities = []
    for i, doc_emb in enumerate(document_embeddings):
        sim = calculate_cosine_similarity(query_embedding, doc_emb)
        similarities.append((sim, documents[i]))

    # 7. Sort documents by similarity in descending order and display the results.
    # The most similar documents are those with the highest cosine similarity score.
    similarities.sort(key=lambda x: x[0], reverse=True)

    print("\n--- En Benzer Belgeler (Sıralı) ---")
    for sim, doc in similarities:
        print(f"Benzerlik Skoru: {sim:.4f} | Belge: '{doc}'")

if __name__ == "__main__":
    main()
