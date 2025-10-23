import os
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from openai import AzureOpenAI

# Only run if this file is executed directly
def execute_chat():
    print("Chat module executed directly.")
    try:
        # connect to the project
        project_endpoint = os.getenv("OPENAI_ENDPOINT")
        if not project_endpoint:
            raise ValueError("OPENAI_ENDPOINT environment variable is not set")
        
        project_client = AIProjectClient(            
                credential=DefaultAzureCredential(),
                endpoint=project_endpoint,
            )
        
        # Get a chat client
        chat_client = project_client.get_openai_client(api_version="2024-10-21")
        
        # Get a chat completion based on a user-provided prompt
        user_prompt = input("Enter a question:")
        
        response = chat_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": user_prompt}
            ]
        )
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    execute_chat()