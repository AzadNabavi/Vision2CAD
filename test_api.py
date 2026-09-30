import anthropic
client = anthropic.Anthropic(api_key="YOUR_API_KEY")
try:
    response = client.messages.create(
        model="claude-opus-5",
        max_tokens=2000,
        messages=[{"role": "user", "content": "Respond with the word test."}]
    )
    print(response.content)
except Exception as e:
    print(f"Error: {e}")
