import mlflow
from mlflow.pyfunc import PythonModel
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from databricks.vector_search.client import VectorSearchClient

class LakehouseRAGModel(PythonModel):
    def load_context(self, context=None):
        self.endpoint_name = "lakehouse_rag_endpoint"
        self.index_name = "ai_catalog.rag_schema.docs_vector_idx"
        self.llm_endpoint = "databricks-meta-llama-3-3-70b-instruct"

    def _retrieve_context(self, query: str, top_k: int = 3) -> str:
        vsc = VectorSearchClient()
        index = vsc.get_index(
            endpoint_name=self.endpoint_name, 
            index_name=self.index_name
        )
        search_res = index.similarity_search(
            query_text=query,
            columns=["chunk_content"],
            num_results=top_k
        )
        data_rows = search_res.get("result", {}).get("data_array", [])
        if not data_rows:
            return ""
        return "\n\n".join([row[0] for row in data_rows])

    def predict(self, context, model_input):
        w = WorkspaceClient()
        
        if isinstance(model_input, pd.DataFrame):
            questions = model_input["question"].tolist()
        elif isinstance(model_input, dict):
            questions = model_input.get("question", [])
            if isinstance(questions, str):
                questions = [questions]
        else:
            questions = [str(model_input)]

        answers = []
        for question in questions:
            retrieved_chunks = self._retrieve_context(question)
            
            system_prompt = (
                "You are an enterprise AI assistant. Answer the user question accurately "
                "based ONLY on the context provided below. If the context does not contain "
                "the necessary information, answer 'I do not have sufficient information in the provided documents.'\n\n"
                f"Context:\n{retrieved_chunks}"
            )

            try:
                response = w.serving_endpoints.query(
                    name=self.llm_endpoint,
                    messages=[
                        ChatMessage(role=ChatMessageRole.SYSTEM, content=system_prompt),
                        ChatMessage(role=ChatMessageRole.USER, content=question)
                    ],
                    max_tokens=300,
                    temperature=0.1
                )
                answers.append(response.choices[0].message.content)
            except Exception as e:
                answers.append(f"LLM Serving Error: {str(e)}")

        return answers

mlflow.models.set_model(LakehouseRAGModel())