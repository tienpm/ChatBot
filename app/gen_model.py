from transformers import pipeline


def llm_model_infer(message):
    if "hello" in message.lower():
        return "Hello there! How can I help you"
    else:
        return "I'm still learning. Perhaps try a different question?"
