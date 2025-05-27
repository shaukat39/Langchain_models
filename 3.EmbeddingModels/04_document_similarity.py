from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)

documents = [
    "Shahid Afridi is an Pakistani cricketer known for his aggressive batting and leadership.",
    "Younas Khan is a former Pakistan captain famous for his calm demeanor and finishing skills.",    
    "Baber Azam is known for his elegant batting.",
    "Waqar  Younas is an Pakistani fast bowler known for his yorkers."
]

query = 'tell me about Waqar Younas'

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("similarity score is:", score)