from sentence_transformers import SentenceTransformer
import os
import numpy as np
import pickle

# Set file paths for loading embeddings
embedding_file = "saved_embeddings.npy"
chunks_file = "saved_chunks.pkl"

# Load embeddings and chunks
if os.path.exists(embedding_file) and os.path.exists(chunks_file):
    print("Loading saved embeddings and chunks...\n\n")
    chunk_embeddings = np.load(embedding_file)  # Load embeddings as binary
    with open(chunks_file, 'rb') as f:          # Load chunks with pickle in binary mode
        chunks = pickle.load(f)
else:
    raise FileNotFoundError("Embeddings or chunks file not found. Run `prepare_embeddings.py` first to generate them.")

# Initialize Cohere client (or other language model API)
import cohere

api_key = 'cxcqLuiWORNXmgZSKdiJEtvtmCvhH7NZsFixUwlp'  # Replace with your Cohere API key
co = cohere.Client(api_key)

# Set the embedding model to a more powerful one from sentence-transformers
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def get_text_embedding(text):
    return embedding_model.encode(text, convert_to_tensor=False)

# Function to query Cohere API
def cohere_llm(prompt, temperature=0.2, max_tokens=100):
    response = co.generate(
        model='command-xlarge-nightly',
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response.generations[0].text.strip()

# Improved prompt with specific instructions
def create_enhanced_prompt(context, query):
    enhanced_prompt = (
        f"Context: {context}\n\n"
        "Based on the context provided above, please provide a concise, accurate, and well-structured answer to the following query.\n\n"
        f"Query: {query}\n\n"
        "Please ensure the response is clear and directly addresses the query while utilizing relevant information from the context.\n\n"
        "Answer in a concise and precise manner:"
    )
    return enhanced_prompt

# Similarity metrics: Cosine, Euclidean, or Inner Product
def compute_similarity(query_embedding, document_embeddings, metric='cosine'):
    if metric == 'cosine':
        similarities = np.dot(document_embeddings, query_embedding) / (
            np.linalg.norm(document_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
    elif metric == 'euclidean':
        similarities = -np.linalg.norm(document_embeddings - query_embedding, axis=1)  # Negative for sorting
    elif metric == 'inner_product':
        similarities = np.dot(document_embeddings, query_embedding)
    else:
        raise ValueError(f"Unsupported similarity metric: {metric}")
    
    return similarities

# Retrieve top N chunks with re-ranking option
def query_engine(query, top_n=3, similarity_metric='cosine'):
    # Generate query embedding
    query_embedding = get_text_embedding(query)

    # Compute similarity between query and chunk embeddings
    similarities = compute_similarity(query_embedding, chunk_embeddings, metric=similarity_metric)

    # Get indices of the top N most similar chunks
    top_n_indices = similarities.argsort()[::-1][:top_n]

    # Retrieve the most relevant chunks
    top_chunks = [chunks[i] for i in top_n_indices]
    
    # Combine the top chunks
    combined_context = "\n\n".join(top_chunks)

    # Create an enhanced prompt with the combined context and query
    enhanced_prompt = create_enhanced_prompt(combined_context, query)

    # Use Cohere for generating the response
    answer = cohere_llm(enhanced_prompt, temperature=0.2, max_tokens=100)

    return answer

# Simple chat function
def chatbot():
    print("Welcome to the Document-based Chatbot! Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Chatbot: Goodbye!")
            break

        # Ask the question to the query engine
        response = query_engine(user_input, top_n=3, similarity_metric='cosine')
        print(f"Chatbot: {response}\n")

# Run the chatbot
chatbot()
