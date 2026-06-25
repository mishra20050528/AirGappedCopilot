import ollama

response = ollama.chat(
    model='phi3',
    messages=[
        {
            'role': 'user',
            'content': '''
CPU=92%
Latency=240ms
Packet Loss=9%

Return output in this format only:

Issue:
Cause:
Action:

Keep answer under 30 words.
'''
        }
    ]
)

print(response['message']['content'])