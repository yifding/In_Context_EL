import openai
from openai import OpenAI

openai.api_key = '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'
client = OpenAI()

response = client.chat.completions.create(
  model="gpt-4-vision-preview",
  messages=[
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this person with celebA attributes in binary classification"},
        {
          "type": "image_url",
          "image_url": {
            "url": "https://s.hdnux.com/photos/51/23/24/10827008/4/1200x0.jpg",
          },
        },
      ],
    }
  ],
  max_tokens=300,
)

print(response.choices[0])