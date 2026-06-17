from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

index = pc.Index(
    os.getenv("PINECONE_INDEX")
)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

df = pd.read_csv(
    "faq_dataset.csv"
)

vectors=[]

for i,row in df.iterrows():

    text=f"""
    Question:
    {row["question"]}

    Answer:
    {row["answer"]}
    """

    embedding=model.encode(
        text
    ).tolist()

    vectors.append(

        {
            "id":str(i),

            "values":embedding,

            "metadata":{

                "question":
                row["question"],

                "answer":
                row["answer"]
            }
        }
    )

index.upsert(
    vectors=vectors
)

print("Vectors uploaded")