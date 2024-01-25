# # Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="liuhaotian/llava-v1.5-7b")

# # Load model directly
# from transformers import AutoProcessor, AutoModelForCausalLM

# processor = AutoProcessor.from_pretrained("liuhaotian/llava-v1.5-7b")
# model = AutoModelForCausalLM.from_pretrained("liuhaotian/llava-v1.5-7b")