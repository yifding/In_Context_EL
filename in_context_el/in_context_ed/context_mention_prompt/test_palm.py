import google.generativeai as palm

completion = palm.generate_text(
...     model='models/text-bison-001',
...     prompt=prompt,
...     temperature=0,
...     # The maximum length of the response
...     max_output_tokens=800,
... )

output = completion.result