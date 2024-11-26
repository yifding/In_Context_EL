import openai
import time
from tqdm import tqdm
from in_context_el.openai_key import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

def openai_chatgpt(prompt, model="gpt-3.5-turbo"):
    openai_output = openai.ChatCompletion.create(
                model=model,
                messages=[
                {"role": "system", "content": prompt},
                ]
            )
    complete_output = openai_output["choices"][0]["message"]['content']
    return complete_output


def openai_completion(prompt, model="text-davinci-003"):
    openai_output = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=100, # default is set to 16, too short
    )
    complete_output = openai_output["choices"][0]['text']
    return complete_output


if __name__ == '__main__':
    from in_context_el.openai_key import OPENAI_API_KEY
    prompt = 'it is just a test'
    openai.api_key = OPENAI_API_KEY
    for _ in tqdm(range(100)):
        time.sleep(1)
        # complete_output = openai_completion(prompt, model='text-curie-001')
        # complete_output = openai_completion(prompt)
        complete_output = openai_chatgpt(prompt)

    