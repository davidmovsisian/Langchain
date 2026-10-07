from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage

model = init_chat_model(
    model="openai:gpt-4o",
    max_tokens=500)

conversation = [
    SystemMessage("You are a helpful assistant that translates English to French."),
    HumanMessage("Translate: I love programming."),
    AIMessage("J'adore la programmation."),
    HumanMessage("Translate: I love building applications.")
]

# response = model.invoke(conversation)
# print(response.content)

# Batch processing example
for response in model.batch_as_completed([
    "How do airplanes fly?",
    "What is quantum computing?"
]):
    print(response)

for index, message in response:
    print(message.content)