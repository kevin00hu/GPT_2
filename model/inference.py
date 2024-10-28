import tiktoken
import torch
import torch.nn.functional as F

MODEL_PATH = "./checkpoints/" + "model.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = torch.load(MODEL_PATH).eval().to(device)
enc = tiktoken.get_encoding("r50k_base")

def chat(user_input:str="", max_token:int=800)->None:
    """ Print the generation output to the console """
    if not user_input:
        return
    input_ids = torch.tensor(enc.encode(user_input)).view(1,-1).to(device)

    for _ in range(max_token):
        output = model(input_ids) # (B, T, C)
        output = F.softmax(output[:, -1, :], dim=-1)
        next_token = torch.multinomial(output, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=-1)

        if next_token == enc.n_vocab-1:
            break

        new_output = enc.decode(input_ids.tolist()[0])
        print(f"\r{new_output}", end="")

def gradio_chat(user_input:str="", max_token:int=800)->str:
    """ Generation function for gradio interface """
    if not user_input:
        return
    input_ids = torch.tensor(enc.encode(user_input)).view(1,-1).to(device)

    for _ in range(max_token):
        output = model(input_ids) # (B, T, C)
        output = F.softmax(output[:, -1, :], dim=-1)
        next_token = torch.multinomial(output, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=-1)

        if next_token == enc.n_vocab-1:
            break

        new_output = enc.decode(input_ids.tolist()[0])

    return new_output

if __name__ == "__main__":
    chat("# High Sensitivity Beamformed")


