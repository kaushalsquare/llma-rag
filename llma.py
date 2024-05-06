import os
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext,StorageContext,load_index_from_storage
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt


os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
basepath = os.getcwd()
filepath = os.path.join(basepath, "data")
PERSIST_DIR = './storage'

'''
    Indexing - Converting documents to Index
    It is first going to convert to Vectors then to Index
    Or this process is called Generating Embedding
'''

if not os.path.exists(PERSIST_DIR):
    # Load the Documents
    documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
    # Create the Index
    indexes = VectorStoreIndex.from_documents(documents,show_progress=True)
    # Store the Index into Storage
    indexes.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)    
    indexes = load_index_from_storage(storage_context)


query_engine = indexes.as_query_engine()

qa = query_engine.query("What is the essay about ?")

print("qa",qa)

# from llama_index.core.retrievers import VectorIndexRetriever
# from llama_index.core.query_engine import RetrieverQueryEngine
# from llama_index.core.indices.postprocessor import SimilarityPostprocessor

# retriever = VectorIndexRetriever(index=indexes,similarity_top_k=5)

# postprocess = SimilarityPostprocessor(similarity_cutoff=0.8)
# query_engine = RetrieverQueryEngine(retriever=retriever,node_postprocessors=[postprocess])

# qa = query_engine.query("What is the essay about ?")

# print(qa)