from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

print("Cuda available: {}".format(torch.cuda.is_available()))
torch.cuda.init()
print("Cuda initialized: {}".format(torch.cuda.is_initialized()))

# Let's chat for 5 lines
for step in range(25):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(
        input(">> User:") + tokenizer.eos_token, return_tensors="pt"
    )

    # append the new user input tokens to the chat history
    bot_input_ids = (
        torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
        if step > 0
        else torch.cat([new_user_input_ids], dim=-1)
    )

    # generated a response while limiting the total chat history to 1000 tokens,
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=200,
        pad_token_id=tokenizer.eos_token_id,
        temperature=1.05,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        no_repeat_ngram_size=5
    )

    # pretty print last ouput tokens from bot
    print(
        "MeidoGPT: {}".format(
            tokenizer.decode(
                chat_history_ids[:, bot_input_ids.shape[-1] :][0],
                skip_special_tokens=True,
            )
        )
    )

