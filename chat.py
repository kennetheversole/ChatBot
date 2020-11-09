import random 
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cpu')


with open('intents.json', 'r+') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()



bot_name = "bot"
print("Let's chat! Type 'quit' to exit")

while True:
    sentence_org = input("You: ")
    if sentence_org == "quit":
        break
    sentence = tokenize(sentence_org)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)

    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand... could you help me? (yes/no)")
        response = input("You: ")
        if response == "yes":
            print(f"What is the subject of the message? example [{tags}]")
            response = input("You: ")
            response.lower()
            if response in tags:
                for intent in intents["intents"]:
                    if response == intent["tag"]:
                        intent["patterns"].append(sentence_org)
                        print(f"{bot_name}: Adding this to my brain, thank you")
f.close()