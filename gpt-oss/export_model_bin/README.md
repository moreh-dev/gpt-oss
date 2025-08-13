# Model weights
- tiny-random/gpt-oss (7M): [Hugging Face](https://huggingface.co/tiny-random/gpt-oss)
- openai/gpt-oss-120b (120B): [Hugging Face](https://huggingface.co/openai/gpt-oss-120b)
- openai/gpt-oss-20b (20B): [Hugging Face](https://huggingface.co/openai/gpt-oss-20b)

# Note
- Run convert_state_dict.py to convert the keys weights to the format accepted by [gpt-oss git repo](https://github.com/openai/gpt-oss/tree/main/gpt_oss)
- Then run generate_bin.py to generate weights + config in .bin format
