# Serverless Lakehouse RAG Pipeline on Azure Databricks

Tech Stack: Azure Databricks, Unity Catalog, Delta Lake (CDF), Databricks Vector Search, MLflow 2.x, Foundation Model Serving (Meta Llama 3), PySpark, Python SDK.

## Executive Summary
Modern enterprise RAG systems often suffer from architectural fragmentation—relying on external vector databases, loose API keys, and disparate embedding pipelines outside the core enterprise data boundary. This project demonstrates an end-to-end, lakehouse-native Retrieval-Augmented Generation (RAG) system built entirely within the Azure Databricks ecosystem, ensuring strict data governance, automated vector synchronization, and serverless compute efficiency.


## System Architecture & Key Components
1.  Unstructured Data Ingestion & Governance:

- Utilized Unity Catalog Volumes (ai_catalog.rag_schema.raw_documents) for secure, audited landing of unstructured enterprise files (PDFs, Markdown, text).

- Implemented a sliding-window text chunker (400-character windows, 40-character overlap) converting unstructured text into structured tabular records.

2. Streaming Delta Lake Storage with CDF:

- Stored chunked text into managed Delta tables (ai_catalog.rag_schema.source_docs).

- Configured Change Data Feed (delta.enableChangeDataFeed = true) to track row-level insert, update, and delete events automatically, eliminating costly full-table re-indexing.

3. Managed Vector Indexing:

- Provisioned a Databricks Vector Search Standard Endpoint paired with a managed Delta Sync index (docs_vector_idx).

- Leveraged built-in embedding pipelines (databricks-bge-large-en), allowing Databricks to manage embedding generation and HNSW indexing without requiring custom vectorization scripts.

4. Foundation Model Integration & Multi-Turn Reasoning:

- Integrated pre-provisioned foundation model endpoints running Meta Llama 3 via the Databricks Workspace Client.

- Constructed conversational grounding prompts that constrain model completions strictly to context retrieved from the lakehouse, complete with conversation history management for contextual multi-turn Q&A.

5. Model Registry & Serverless Serving:

- Packaged the retrieval engine into an MLflow custom pyfunc model utilizing models-from-code logging patterns to eliminate unpickleable network client references.

- Registered models into Unity Catalog's three-level namespace and assigned production deployment aliases (champion).

- Configured real-time REST API serving via Databricks Serverless Model Serving with scale-to-zero scaling rules.





