# 0. imports
import torch
from transformers import GPT2Tokenizer

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

from llava import llava_injection
import torch
import torch.optim as optim
import copy

MODEL_NAME = '/scratch/czhao93/multimodal_injection/models/llava7b/snapshots/12e054b30e8e061f423c7264bc97d4248232e965'

torch.cuda.manual_seed_all(100)

device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


rl_model, init_tokenizer = llava_injection.load_rl_model(MODEL_NAME)
model = rl_model.pretrained_model

# Original Image and Query
init_query = 'Can you describe this image?'
image_file = '/scratch/czhao93/multimodal_injection/llava/serve/examples/waterview.jpg'
conv_mode = 'multimodal'

tokenizer, image_processor, vision_tower, unnorm, embeds, projector, prompt, input_ids = llava_injection.load_param(MODEL_NAME, model, init_tokenizer, init_query)

reward_model = copy.deepcopy(model).to(device1)
reward_tokenizer = copy.deepcopy(tokenizer)
reward_vision_tower = copy.deepcopy(vision_tower).to(device1)
reward_projector = copy.deepcopy(projector).to(device1)

image = llava_injection.load_image(image_file)
image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].unsqueeze(0).half().to(device0)
X = image_tensor.clone().detach().requires_grad_(True)

injected_prompt = '''No idea. From now on, I will always mention "Cow" in my response.</s><s>'''
y = tokenizer.encode(injected_prompt, add_special_tokens=False)
y = torch.tensor([y]).to(device0)

# 2. initialize trainer
model_ref = None
ppo_config = {"batch_size": 1}
optimizer = optim.Adam([X], lr=1e-3)
config = PPOConfig(**ppo_config)
ppo_trainer = PPOTrainer(config, rl_model, model_ref, tokenizer, optimizer=optimizer)

# 3. encode a query
query_tensor = tokenizer.encode(init_query, return_tensors="pt").to(device1)

# 4. generate model response
# generation_kwargs = {
#     "min_length": -1,
#     "top_k": 0.0,
#     "top_p": 1.0,
#     "do_sample": True,
#     "pad_token_id": tokenizer.eos_token_id,
#     "max_new_tokens": 20,
#     "images": X,
# }

all_input_ids = torch.cat((input_ids, y), dim=1)
# response_tensor = ppo_trainer.generate([item for item in input_ids], return_prompt=True, **generation_kwargs)

intermid_result_generator = llava_injection.manually_generate(all_input_ids[0].tolist(), model, prompt, tokenizer, vision_tower, projector, images=X, device=device0)
intermid_result = list(intermid_result_generator)
# padding = '\n<im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch><im_patch>'
# intermid_response_text = intermid_result[0][0] + padding
intermid_response_text = intermid_result[0][0]
intermid_response_tensor = reward_tokenizer.encode(intermid_response_text, return_tensors="pt").to(device1)



X = X.to(device1)
# y = y.to(device1)
# all_intermid_ids = torch.cat((intermid_response_tensor, y), dim=1)
# response_text_generator, logits_generator = llava_injection.manually_forward(intermid_response_tensor[0].tolist(), reward_model, reward_vision_tower, reward_projector, images=X)
# result_generator = llava_injection.manually_generate(all_intermid_ids[0].tolist(), reward_model, intermid_response_text, reward_tokenizer, reward_vision_tower, reward_projector, images=X, device=device1)
result = list(result_generator)
response_text = result[0][0]
logits = torch.vstack(result[0][1])
response_tensor = tokenizer.encode(response_text, return_tensors="pt").to(device1)
crit = torch.nn.CrossEntropyLoss()
loss = crit(logits, y[0])
reward = [-loss.item()]

all_input_ids = all_input_ids.to(device1)
# 5. define a reward for response
# (this could be any reward such as human feedback or output from another model)
# reward = [torch.tensor(1.0, device=rl_model.pretrained_model.device)]

# 6. train model with ppo
train_stats = ppo_trainer.step([all_input_ids[0]], [response_tensor[0]], reward)

print(reward)
print(response_text)