import os
from transformers import pipeline, set_seed, Conversation

os.environ['TRANSFORMERS_CACHE'] = '/nfs/yding4/transformers_cache'


# generator = pipeline('text-generation', model='facebook/opt-13b')
# generator = pipeline('text-generation', model='gpt2')
generator = pipeline('text-generation', model='meta-llama/Llama-2-7b-chat-hf', token="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")
# generator = pipeline('text-generation', model='meta-llama/Llama-2-70b-chat-hf', token="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")


set_seed(42)
s = generator(
    "The Ministry of Defense refers to the governmental organization responsible for overseeing and directing the armed forces of Iran.\n\nWhich of the following entities is Ministry of Defense in this sentence?\n\n(1). Ministry of Defense (Israel) Ministry of Defense (Israel)\n\nThe Ministry of Defense (, \"Misrad HaBitahon\") () of the government of Israel, is the governmental department responsibl\n(2). Secretariat of National Defense (Mexico) Secretariat of National Defense (Mexico)\n\nThe Mexican Secretariat of National Defense (, \"SEDENA\") is the government department responsible for managi\n(3). Ministry of Defense (Japan) Ministry of Defense (Japan)\n\nThe is a cabinet-level ministry of the Government of Japan. It is headquartered in Shinjuku, Tokyo, and is the largest or\n(4). Ministry of Defence (United Kingdom) Ministry of Defence (United Kingdom)\n\nThe Ministry of Defence (MOD) is the British government department responsible for implementing the defence poli\n(5). Ministry of Defense (Argentina) Ministry of Defense (Argentina)\n\nThe Ministry of Defense of Argentina is a ministry of the national executive power that deals with everything related\n(6). Ministry of Defence (Somalia) Ministry of Defence (Somalia)\n\nThe Ministry of Defence of the Federal Government of Somalia (FGS) is the government body in charge of the Somali Armed\n(7). Ministry of Defence (Russia) Ministry of Defence (Russia)\n\nThe Ministry of Defence of the Russian Federation (, informally abbreviated as \u041c\u0438\u043d\u043e\u0431\u043e\u0440\u043e\u043d\u043f\u0440\u043e\u043c) exercises administrative a\n(8). Ministry of Defense (Peru) Ministry of Defense (Peru)\n\nThe Ministry of Defence of Peru () is the agency of the Peruvian government responsible for safeguarding of national secur\n(9). Ministry of National Defense (Romania) Ministry of National Defense (Romania)\n\nThe Ministry of National Defense () is one of the fifteen ministries of the Government of Romania.\nMinistry.\nT\n(10). Ministry of Defence and Urban Development Ministry of Defence and Urban Development\n\nThe Ministry of Defence and Urban Development is the Sri Lankan government ministry responsible for impleme\n", 
    # max_length=512,
    # pad_token_id = 50256, # 
)

# conversation = Conversation("The Ministry of Defense refers to the governmental organization responsible for overseeing and directing the armed forces of Iran.\n\nWhich of the following entities is Ministry of Defense in this sentence?\n\n(1). Ministry of Defense (Israel) Ministry of Defense (Israel)\n\nThe Ministry of Defense (, \"Misrad HaBitahon\") () of the government of Israel, is the governmental department responsibl\n(2). Secretariat of National Defense (Mexico) Secretariat of National Defense (Mexico)\n\nThe Mexican Secretariat of National Defense (, \"SEDENA\") is the government department responsible for managi\n(3). Ministry of Defense (Japan) Ministry of Defense (Japan)\n\nThe is a cabinet-level ministry of the Government of Japan. It is headquartered in Shinjuku, Tokyo, and is the largest or\n(4). Ministry of Defence (United Kingdom) Ministry of Defence (United Kingdom)\n\nThe Ministry of Defence (MOD) is the British government department responsible for implementing the defence poli\n(5). Ministry of Defense (Argentina) Ministry of Defense (Argentina)\n\nThe Ministry of Defense of Argentina is a ministry of the national executive power that deals with everything related\n(6). Ministry of Defence (Somalia) Ministry of Defence (Somalia)\n\nThe Ministry of Defence of the Federal Government of Somalia (FGS) is the government body in charge of the Somali Armed\n(7). Ministry of Defence (Russia) Ministry of Defence (Russia)\n\nThe Ministry of Defence of the Russian Federation (, informally abbreviated as \u041c\u0438\u043d\u043e\u0431\u043e\u0440\u043e\u043d\u043f\u0440\u043e\u043c) exercises administrative a\n(8). Ministry of Defense (Peru) Ministry of Defense (Peru)\n\nThe Ministry of Defence of Peru () is the agency of the Peruvian government responsible for safeguarding of national secur\n(9). Ministry of National Defense (Romania) Ministry of National Defense (Romania)\n\nThe Ministry of National Defense () is one of the fifteen ministries of the Government of Romania.\nMinistry.\nT\n(10). Ministry of Defence and Urban Development Ministry of Defence and Urban Development\n\nThe Ministry of Defence and Urban Development is the Sri Lankan government ministry responsible for impleme\n", 
# )
# s = generator(conversation)
# print(s.generated_responses)
print(s)
# import requests

# API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
# API_TOKEN = "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"
# headers = {"Authorization": f"Bearer {API_TOKEN}"}

# def query(payload):
# 	response = requests.post(API_URL, headers=headers, json=payload)
# 	return response.json()
	
# output = query({
# 	"inputs": "Can you please let us know more details about your ",
# })

# print(output)