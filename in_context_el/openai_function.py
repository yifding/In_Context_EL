import openai
from tqdm import tqdm

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
    )
    complete_output = openai_output["choices"][0]['text']
    return complete_output


if __name__ == '__main__':
    from in_context_el.openai_key import OPENAI_API_KEY
    prompt = 'it is just a test'
    openai.api_key = OPENAI_API_KEY
    for _ in tqdm(range(100)):
        complete_output = openai_completion(prompt)
    