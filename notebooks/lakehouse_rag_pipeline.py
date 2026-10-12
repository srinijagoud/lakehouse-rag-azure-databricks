# Unity Catalog Setup
spark.sql("CREATE CATALOG IF NOT EXISTS ai_catalog")
spark.sql("USE CATALOG ai_catalog")
spark.sql("CREATE SCHEMA IF NOT EXISTS rag_schema")
spark.sql("USE SCHEMA rag_schema")
spark.sql("CREATE VOLUME IF NOT EXISTS ai_catalog.rag_schema.raw_documents")

# Chunking & Delta Lake Ingestion with CDF
import os, uuid, pypdf
from pyspark.sql.types import StructType, StructField, StringType

volume_path = "/Volumes/ai_catalog/rag_schema/raw_documents"
os.makedirs(volume_path, exist_ok=True)

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 40) -> list:
    chunks, start = [], 0
    clean_text = " ".join(text.split())
    while start < len(clean_text):
        chunks.append(clean_text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks

records = []
for file_name in os.listdir(volume_path):
    full_path = os.path.join(volume_path, file_name)
    extracted_text = ""
    if file_name.endswith(".pdf"):
        reader = pypdf.PdfReader(full_path)
        for page in reader.pages:
            extracted_text += page.extract_text() or ""
    elif file_name.endswith((".txt", ".md")):
        with open(full_path, "r", encoding="utf-8") as f:
            extracted_text = f.read()

    if extracted_text:
        splits = chunk_text(extracted_text)
        for idx, segment in enumerate(splits):
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{file_name}_{idx}"))
            records.append((chunk_id, file_name, segment))

schema = StructType([
    StructField("id", StringType(), False),
    StructField("doc_name", StringType(), True),
    StructField("chunk_content", StringType(), True)
])

df_chunks = spark.createDataFrame(records, schema=schema)
delta_table_name = "ai_catalog.rag_schema.source_docs"

df_chunks.write.format("delta").mode("overwrite").option("delta.enableChangeDataFeed", "true").saveAsTable(delta_table_name)
spark.sql(f"ALTER TABLE {delta_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

# Vector Search Index Creation
import time
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()
endpoint_name = "lakehouse_rag_endpoint"
index_name = "ai_catalog.rag_schema.docs_vector_idx"

if endpoint_name not in [e.get("name") for e in vsc.list_endpoints().get("endpoints", [])]:
    vsc.create_endpoint(name=endpoint_name, endpoint_type="STANDARD")

while vsc.get_endpoint(endpoint_name).get("endpoint_status", {}).get("state") != "ONLINE":
    time.sleep(15)

try:
    vsc.create_delta_sync_index(
        endpoint_name=endpoint_name,
        index_name=index_name,
        source_table_name=delta_table_name,
        pipeline_type="TRIGGERED",
        primary_key="id",
        embedding_source_column="chunk_content",
        embedding_model_endpoint_name="databricks-bge-large-en"
    )
except Exception as e:
    print(f"Index notice: {e}")