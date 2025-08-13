import tiktoken

enc = tiktoken.get_encoding("o200k_harmony")
sentence = "🍣 sushi and 🍜 ramen"
tokens = enc.encode(sentence)
print("Token IDs:", tokens)
decoded = enc.decode(tokens)
print("Decoded sentence:", decoded)
