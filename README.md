**Simple Retrieval Augmented Generation Service**

*vector_search.py*


The file first instantiates the _all-mpnet-base-v2_ sentence embedding model and creates embeddings using chunks from the context text.
These chunks are collections of N sentences from the context text. 
We utilize the faiss model indexed for dot product similarity for computing cosine similarity between the normalized query embeddings and the sentence embeddings.
Our vector search is also able to retrieve the top N most similar chunks of context text to a user's query.

*answer.py*


This file instantiates our vector_search class and retrieves the top N most similar sentence chunks from the context document. Using these retrieved chunks of relevant texts, our prompt is augmented 
and then fed into the __gemini-it__ model which generates an answer to the user's prompt. Effectively this is a simple RAG system. 
