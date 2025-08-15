import tiktoken

enc = tiktoken.get_encoding("o200k_harmony")
sentence = "🍣 sushi and 🍜 ramen"
tokens = enc.encode(sentence)
tokens = [10, 16, 28, 17, 11, 220, 16, 28, 17, 28, 18 , 11, 220]
print("Token IDs:", type(tokens))
decoded = enc.decode(tokens)
print("Decoded sentence:", decoded)
