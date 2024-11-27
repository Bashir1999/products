import gradio as gr
from huggingface_hub import InferenceClient

# --- Model 1: AI Chatbot Setup ---
client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.3")

# Personalities for AI Chatbot
PERSONALITIES = {
    "Friendly": "You are a friendly and helpful assistant.",
    "Professional": "You are a professional and concise assistant.",
    "Humorous": "You are a witty and humorous assistant.",
    "Empathetic": "You are a compassionate and empathetic assistant."
}

# Chatbot Functions
def respond(message, history, personality):
    system_message = PERSONALITIES[personality]
    messages = [{"role": "system", "content": system_message}]

    for user_message, bot_message in history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": bot_message})

    messages.append({"role": "user", "content": message})
    response = client.chat_completion(messages, max_tokens=1024)
    bot_message = response["choices"][0]["message"]["content"]
    history.append((message, bot_message))
    return history, ""

def generate_fun_fact(history):
    message = "Give me a fun fact."
    system_message = "You are a helpful assistant that shares fun facts when asked."
    messages = [{"role": "system", "content": system_message}]

    for user_message, bot_message in history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": bot_message})

    messages.append({"role": "user", "content": message})
    response = client.chat_completion(messages, max_tokens=256)
    fun_fact = response["choices"][0]["message"]["content"]
    history.append((message, fun_fact))
    return history

def generate_daily_challenge(history):
    message = "Give me a daily challenge."
    system_message = "You are a helpful assistant that gives fun or motivational daily challenges."
    messages = [{"role": "system", "content": system_message}]

    for user_message, bot_message in history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": bot_message})

    messages.append({"role": "user", "content": message})
    response = client.chat_completion(messages, max_tokens=256)
    challenge = response["choices"][0]["message"]["content"]
    history.append((message, challenge))
    return history

def generate_inspiration(history):
    message = "Give me an inspirational quote or motivational message."
    system_message = "You are a helpful assistant that provides inspiring or motivational quotes when asked."
    messages = [{"role": "system", "content": system_message}]

    for user_message, bot_message in history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": bot_message})

    messages.append({"role": "user", "content": message})
    response = client.chat_completion(messages, max_tokens=256)
    inspiration = response["choices"][0]["message"]["content"]
    history.append((message, inspiration))
    return history

def clear_conversation():
    return [], ""

# Custom CSS
custom_css = """
body {
    background-color: #080505;
}
.gradio-container {
    max-width: 900px;
    margin: auto;
    background-color: #ffffff;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 16px rgba(0, 0, 0, 0.2);
}
"""

# --- Gradio Interface ---
with gr.Blocks(css=custom_css) as chatbot_interface:
    gr.Markdown("### AI Chatbot - Choose a personality and start chatting")
    personality = gr.Radio(choices=["Friendly", "Professional", "Humorous", "Empathetic"], value="Friendly", label="Personality")
    chatbot = gr.Chatbot(label="Chatbot", height=300)
    message = gr.Textbox(placeholder="Type your message here...")
    history = gr.State([])
    send_btn = gr.Button("Send")
    clear_btn = gr.Button("Clear")
    fun_fact_btn = gr.Button("Fun Fact")
    inspire_me_btn = gr.Button("Inspire Me")
    challenge_btn = gr.Button("Daily Challenge")
    send_btn.click(respond, inputs=[message, history, personality], outputs=[chatbot, message])
    clear_btn.click(clear_conversation, outputs=[chatbot, message])
    fun_fact_btn.click(generate_fun_fact, inputs=history, outputs=chatbot)
    challenge_btn.click(generate_daily_challenge, inputs=history, outputs=chatbot)
    inspire_me_btn.click(generate_inspiration, inputs=history, outputs=chatbot)

# Launch the app
chatbot_interface.launch(share=True)
