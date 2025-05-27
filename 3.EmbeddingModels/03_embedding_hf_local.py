from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Tehran is the capital of Iran",
    "Islamabad is the capital of Pakistan",
    "Paris is the capital of France"
]

vector = embedding.embed_documents(documents)

print(str(vector))