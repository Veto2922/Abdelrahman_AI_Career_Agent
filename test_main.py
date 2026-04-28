from src.Data_ingestion_block.data_ingestion import DataIngestion


data_ingestion = DataIngestion()

data_ingestion.upload_file("data/Abdelrahman_Mohamed_CV_14.9.pdf")


from src.retrieval_block.tree_retrieval import TreeRetrieval
from pageindex import PageIndexClient
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("Page_index_api")

if not api_key:
    raise ValueError("PageIndex API key is missing.")

pi_client = PageIndexClient(api_key=api_key)


tree_retrieval = TreeRetrieval(pi_client)

print(tree_retrieval.get_docs_titles())

print(tree_retrieval.get_toc([0]))

print(tree_retrieval.retrieve([{"doc_index": 0, "target_ids": ["0007"]}]))


########### test graph #############################################

from src.graph_block.graph import compile_graph
from src.retrieval_block.tree_retrieval import TreeRetrieval
from pageindex import PageIndexClient
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("Page_index_api")

if not api_key:
    raise ValueError("PageIndex API key is missing.")

pi_client = PageIndexClient(api_key=api_key)


tree_retrieval = TreeRetrieval(pi_client)

agent = compile_graph(tree_retrieval)

docs_titles = tree_retrieval.get_docs_titles()

config = {"configurable": {"thread_id": "1"}}

res = agent.invoke(
    {"user_query": "ايه هي المشاريع الي اشتغلت عليها؟", "docs_titles": docs_titles},
    config,
)

for m in res["messages"]:
    m.pretty_print()
