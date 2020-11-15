from fastapi import FastAPI
from pydantic import BaseModel

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datetime import datetime, timedelta

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

print("Cuda available: {}".format(torch.cuda.is_available()))
torch.cuda.init()
print("Cuda initialized: {}".format(torch.cuda.is_initialized()))

app = FastAPI()

class Message(BaseModel):
    message: str

chat_history = torch.tensor([], dtype=torch.int64)
messages_received = 0
last_message_received = datetime.utcnow()

@app.post("/", response_model=Message)
def get_response(msg: Message):
    global chat_history, messages_received, last_message_received

    # Reset the state if conditions are met
    if messages_received > 5 or (datetime.utcnow() - last_message_received) > timedelta(minutes=5):
        chat_history = torch.tensor([], dtype=torch.int64)
        messages_received = 0
        last_message_received = datetime.utcnow()

    new_chat_message = tokenizer.encode(msg.message + tokenizer.eos_token, return_tensors="pt")
    chat_history = torch.cat([chat_history, new_chat_message], dim=-1)
    chat_history_with_response = model.generate(
        chat_history,
        max_length=2000,
        pad_token_id=tokenizer.eos_token_id,
        temperature=1.05,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        no_repeat_ngram_size=5
    )

    response =  {"message": tokenizer.decode(chat_history_with_response[:, chat_history.shape[-1] :][0], skip_special_tokens=True)}

    chat_history = chat_history_with_response

    messages_received += 1

    return response