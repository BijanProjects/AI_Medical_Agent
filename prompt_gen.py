from retriever import similarity_search

context = similarity_search(['Whenever I jump my leg hurts; what should I do?'], 1)
print(context)




# tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
# model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", device_map='auto')


prompt = """Instruction:
You are a specialized AI-powered medical agent trained to provide **clear, structured, and evidence-based** answers to medical questions. Follow these strict rules:

- **Only answer medical-related questions.** If the user asks a non-medical question, respond with: "I am an AI-powered medical agent and cannot help you with non-medical questions!"
- **Use ONLY the provided context** to generate your response. **DO NOT make up any medical facts** that are not explicitly stated in the context.
- **If information is missing in the context**, state: "I do not have enough information to answer this. Please consult a medical professional."
- **Do not repeat information unnecessarily.**
- **If the user asks multiple questions**, structure the response in a step-by-step format.
- **Do not mix unrelated medical terms**; only mention medications or treatments that are supported by the provided context.

Context: 
{}

Question: 
{}

Response:
"""



# query = ['What medications and dosage guide must a pre-diabetic person follow? How much exercise would be good for them?']

# context = similarity_search(query, top_k = 3)
#prompt = prompt.format(context, query[0])



# def generate_response(prompt):
#     tokenized = tokenizer(prompt, return_tensors='pt')
#     output = model.generate(**tokenized, max_length= 2000)
#     return tokenizer.decode(output, skip_special_tokens=True)



# response = generate_response(prompt=prompt)
# print("AI response is:{response}")

# print(f"context 1: {context[0]} \n\n\n\n context 2: {context[1]} \n\n\n\n context 3: {context[2]}")
