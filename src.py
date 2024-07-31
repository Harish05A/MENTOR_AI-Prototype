import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

mental_health_responses = {
    "sad": [
        "I'm sorry to hear that you're feeling sad. It's okay to feel this way. Do you want to talk about what's bothering you?",
        "It sounds like you're going through a tough time. I'm here to listen if you need to talk.",
        "Feeling sad is a natural part of life. Sometimes sharing your feelings can help. What's on your mind?"
    ],
    "anxious": [
        "Anxiety can be overwhelming. Try to take deep breaths and focus on the present moment. What's making you feel anxious?",
        "It's okay to feel anxious. Sometimes talking about it can help. Do you want to share what's on your mind?",
        "I understand that anxiety can be tough to deal with. I'm here to support you. What's causing your anxiety?"
    ],
    "stressed": [
        "Stress can be really challenging. Remember to take breaks and take care of yourself. What's causing your stress?",
        "It's important to manage stress. Talking about it can help. Do you want to share what's stressing you out?",
        "Feeling stressed is common, but it's important to find ways to relax. What's been stressing you out lately?"
    ]
}

def detect_emotion(input_text):

    if any(word in input_text.lower() for word in ["sad", "unhappy", "depressed"]):
        return "sad"
    elif any(word in input_text.lower() for word in ["anxious", "nervous", "worried"]):
        return "anxious"
    elif any(word in input_text.lower() for word in ["stressed", "overwhelmed", "tense"]):
        return "stressed"
    else:
        return None

def generate_response(input_text):
    emotion = detect_emotion(input_text)
    if emotion:
        return random.choice(mental_health_responses[emotion])
    else:
        input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')
        chatbot_output = model.generate(input_ids=input_ids, max_length=50, do_sample=True, top_p=0.95, top_k=50)
        chatbot_response = tokenizer.decode(chatbot_output[0], skip_special_tokens=True)
        return chatbot_response
def chat():
    print("Mentor AI: Hello! How can I help you today? (type 'exit' to end the conversation)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Mentor AI: Take care! Goodbye!")
            break
        chatbot_response = generate_response(user_input)
        print("Mentor AI: " + chatbot_response)


if _name_ == "__main__":
    chat()
