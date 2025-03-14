import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

medical_data = pd.read_csv('train.csv')

# print(len(medical_data))
medical_data.dropna()
# print(len(medical_data))
# print(medical_data.columns)

medical_data['text'] = 'The Question: ' + medical_data['Question'] + '\nExpert response: ' + medical_data['Answer']


# print(medical_data['text'][10])

model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(medical_data['text'].to_list(), show_progress_bar=True)

embeddings = np.array(embeddings).astype('float32')

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, "Medical_QA.index")
medical_data.to_csv("Medical_QA.csv", index=False)

