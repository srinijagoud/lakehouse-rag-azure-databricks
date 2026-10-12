import mlflow
from mlflow import MlflowClient
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput, ServingModelWorkloadType

w = WorkspaceClient()
client = MlflowClient()
mlflow.set_registry_uri("databricks-uc")

model_name = "ai_catalog.rag_schema.lakehouse_rag_service"
endpoint_name = "lakehouse-rag-rest-api"

# Registering Code-Based Model to Unity Catalog
test_query_df = pd.DataFrame({"question": ["What is CDF?"]})
with mlflow.start_run(run_name="Lakehouse_RAG_Production"):
    signature = mlflow.models.infer_signature(test_query_df, pd.Series(["sample output"]))
    model_info = mlflow.pyfunc.log_model(
        artifact_path="rag_agent",
        python_model="models/rag_model_code.py",
        signature=signature,
        pip_requirements=["databricks-sdk", "databricks-vectorsearch", "pandas", "mlflow"],
        registered_model_name=model_name
    )

# Assigning Champion Alias
versions = [int(v.version) for v in w.model_versions.list(full_name=model_name)]
latest_version = str(max(versions))
client.set_registered_model_alias(model_name, "champion", version=latest_version)

# Deploying Serverless REST Endpoint
endpoint_config = EndpointCoreConfigInput(
    served_entities=[
        ServedEntityInput(
            entity_name=model_name,
            entity_version=latest_version,
            workload_size="Small",
            workload_type=ServingModelWorkloadType.CPU,
            scale_to_zero_enabled=True
        )
    ]
)

if endpoint_name not in [e.name for e in w.serving_endpoints.list()]:
    w.serving_endpoints.create(name=endpoint_name, config=endpoint_config)
else:
    w.serving_endpoints.update_config(name=endpoint_name, served_entities=endpoint_config.served_entities)