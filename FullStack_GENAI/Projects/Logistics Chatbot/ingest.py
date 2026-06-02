import os
import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def load_csv():

    data_folder="data"

    shipments=pd.read_csv(
        os.path.join(
            data_folder,
            "shipments.csv"
        )
    )

    orders=pd.read_csv(
        os.path.join(
            data_folder,
            "orders.csv"
        )
    )

    tickets=pd.read_csv(
        os.path.join(
            data_folder,
            "customer_support_tickets.csv"
        )
    )

    docs=[]

    order_ids=set(
        shipments["OrderID"]
    ).union(
        orders["OrderID"]
    )

    for order_id in order_ids:

        text=[]

        order_rows=orders[
            orders["OrderID"]==order_id
        ]

        shipment_rows=shipments[
            shipments["OrderID"]==order_id
        ]

        ticket_rows=tickets[
            tickets["OrderID"]==order_id
        ]

        for _,row in order_rows.iterrows():

            text.append(
                "ORDER -> " +
                " | ".join(
                    [f"{c}:{row[c]}" for c in order_rows.columns]
                )
            )

        for _,row in shipment_rows.iterrows():

            text.append(
                "SHIPMENT -> " +
                " | ".join(
                    [f"{c}:{row[c]}" for c in shipment_rows.columns]
                )
            )

        for _,row in ticket_rows.iterrows():

            text.append(
                "TICKET -> " +
                " | ".join(
                    [f"{c}:{row[c]}" for c in ticket_rows.columns]
                )
            )

        docs.append(

            Document(

                page_content="\n".join(text),

                metadata={

                    "order_id":order_id

                }

            )

        )

    return docs

docs=load_csv()

embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db=Chroma.from_documents(
        docs,
        embeddings,
        persist_directory="vectorstore"
)

db.persist()

print("done")