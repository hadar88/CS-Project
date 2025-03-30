import requests

SERVER_URL = "https://simple-server-6wry.onrender.com"

def greet(name):
    """Sends a greeting to the server."""
    response = requests.get(f"{SERVER_URL}/greet?name={name}")
    return response.text

print(greet("Alice"))  # Example usage, replace "Alice" with any name you want to greet
