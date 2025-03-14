import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


index_data = faiss.read_index('Medical_QA.index')
meta_data = pd.read_csv('Medical_QA.csv')


def similarity_search(query, top_k = 3):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedings = model.encode(query).astype('float32')
    distance, indices = index_data.search(query_embedings, top_k)
    knowledge = []
    for i in indices[0]:
        knowledge.append(meta_data.iloc[i]['text'])
    return knowledge


# print(similarity_search(['Whenever I jump my leg hurts; what should I do?']))


