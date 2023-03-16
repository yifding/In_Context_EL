import openai

def openai_chatgpt(prompt, model="gpt-3.5-turbo"):
    openai_output = openai.ChatCompletion.create(
                model=model,
                messages=[
                {"role": "system", "content": prompt},
                ]
            )
    complete_output = openai_output["choices"][0]["message"]['content']
    return complete_output