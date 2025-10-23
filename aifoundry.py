import os
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

def list_connections():
    try:
        # Get project client
        project_endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
        if not project_endpoint:
            raise ValueError("AZURE_AI_PROJECT_ENDPOINT environment variable is not set")
        
        project_client = AIProjectClient(            
        
                credential=DefaultAzureCredential(),
                endpoint=project_endpoint,
            )
        
        ## List all connections in the project
        connections = project_client.connections
        print("List all connections:")
        for connection in connections.list():
            print(f"{connection.name} ({connection.type})")
            print(f"  creds: {connection.credentials}")

            my_connection = connections.get(name=connection.name)
            print(f"{my_connection.name} ({my_connection.type})")
            print(f"  creds: {my_connection.credentials}")
            
    except Exception as ex:
        print(ex)

# Only run if this file is executed directly
if __name__ == "__main__":
    list_connections()