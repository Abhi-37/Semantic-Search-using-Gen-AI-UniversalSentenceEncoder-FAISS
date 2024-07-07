# Semantic-Search-using-Gen-AI

It reflects the main purpose of the code, which is to perform semantic search on a dataset of text documents using FAISS for indexing and the Universal Sentence Encoder for generating embeddings.

In this code:

We fetch the 20 Newsgroups dataset, a collection of documents spanning various topics.

We preprocess each document by removing email headers, addresses, punctuations, and numbers, and convert text to lowercase for uniformity.

We utilize the Universal Sentence Encoder to generate embeddings, converting each document into a fixed-length numerical representation capturing its semantic meaning.

We construct a FAISS index, a fast similarity search library, and add the document embeddings to enable efficient similarity search.

We define a search function that preprocesses user queries, generates embeddings, and retrieves the most similar documents from the index.

We demonstrate the functionality with an example query ("motorcycle"), displaying the top results ranked by similarity.
