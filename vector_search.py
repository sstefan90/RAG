import numpy as np
import faiss
import pickle
import pymupdf
import nltk
import torch
import pandas as pd
import os

from sentence_transformers import SentenceTransformer
import torch.nn.functional as f
nltk.download('punkt_tab')


class VectorSearch:

    def unpickle_file(self):
        with open('data/df.pickle', 'rb') as file:
            return pickle.load(file)


    def __init__(self):

        self.data = self.unpickle_file()
        self.index = faiss.IndexFlatIP(768)
        self.index.add(np.array(self.data['vectors'].tolist()))

    def find_top_N(self, query_vector, N, verbose=False):
        D, I = self.index.search(query_vector, N)
        if verbose:
            print("distance", D)
            print("index", I)
        return self.data.iloc[I[0]]['text'].tolist()


def read_pdf(file):
    texts = []
    doc = pymupdf.open(file)
    for page in doc:
        text = page.get_text()
        text = text.replace("\n", " ")
        sentences = nltk.sent_tokenize(text)
        texts += sentences
    return texts


def pickle_data(data):
    with open('data/df.pickle', 'wb') as file:
        pickle.dump(data, file)


def process_data(file):
    #process data
    texts = read_pdf(file)

    #create chunks
    chunks = []
    for i in range(0, len(texts), 10):
        if i+10 > len(texts):
            chunks.append(" ".join(texts[i:]))
        else:
            chunks.append(" ".join(texts[i:i+10]))

    #sentence embeddings
    embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device="cpu")
    print("Creating embeddings...")
    embeddings = embedding_model.encode(chunks, batch_size=4, show_progress_bar=True, convert_to_tensor=True)
    embeddings = f.normalize(embeddings, p=2, dim=0)

    emb_list = embeddings.tolist()

    #create dataframe
    df = pd.DataFrame(columns=['text', 'vectors'])
    df['vectors'] = embeddings.tolist()
    df['text'] = chunks

    #pickle data
    pickle_data(df)

    
if __name__ == "__main__":
    '''
    Just for testing!
    '''
    if not os.path.exists('data/df.pickle'):
        #pass
        process_data("data/Nutrition_Doc.pdf")
    
    vs = VectorSearch()
    query = "What are healthy food groups?"
    embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device="cpu")
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    query_embedding = query_embedding.reshape(1, 768)
    query_embedding = f.normalize(query_embedding, p=2, dim=0)
    result = vs.find_top_N(query_embedding, 5, verbose=True)
    print(result)
    
    




        

        
        

