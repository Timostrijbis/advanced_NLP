from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from huggingface_hub import login


# login(token="yourtoken")

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("Number of GPUs:", torch.cuda.device_count())
    print("Current GPU:",
          torch.cuda.get_device_name(torch.cuda.current_device()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
# or "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    device_map="auto",
    trust_remote_code=True
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "¿Cuál es la capital de Francia?"
response = pipe(prompt, max_new_tokens=128, temperature=0.6)

print(response[0]['generated_text'])
