from retriever import similarity_search
from unsloth import FastLanguageModel
from unsloth import FastLlamaModel
import torch



prompt = """Instruction:
You are a specialized AI-powered medical agent trained to provide **clear, structured, and evidence-based** answers to medical query at the end. Follow these strict rules:

- **Only answer medical-related queries.** If the user asks a non-medical question, respond with: "I am an AI-powered medical agent and cannot help you with non-medical questions!"
- **Use ONLY the provided context (questions and answers pairs)** to generate your response. **DO NOT make up any medical facts** that are not explicitly stated in the context.
- **If information is missing in the context**, state: "I do not have enough information to answer this. Please consult a medical professional."
- **Do not repeat information unnecessarily.**
- **If the user asks multiple questions**, structure the response in a step-by-step format answering each question accordingly.
- **Do not mix unrelated medical terms**; only mention medications or treatments that are supported by the provided context.

Context: 
{}

Query: 
{}

Response:
"""


model, tokenizer = FastLlamaModel.from_pretrained(model_name='unsloth/Llama-3.2-3B-Instruct',
                                                  max_seq_length=8192,
                                                  load_in_4bit=True)


def generate(model, tokenizer, query, prompt):
  context = similarity_search(query, top_k = 3)
  prompt = prompt.format(context, query[0])
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
  inputs = tokenizer(prompt, return_tensors='pt').to(device)

  output = model.generate(**inputs, max_new_tokens = 1000, temperature=0.5)
  output = tokenizer.decode(output[0], skip_special_tokens=True)
  response_text = output.split("Response:\n", 1)[1] if "Response:\n" in output else output.strip()
    
  return response_text.strip()



query = input("Please ask your medical question:\n")

print(generate(model, tokenizer, [query], prompt))
