import os
import replicate
os.environ['REPLICATE_API_TOKEN'] = 'r8_EzgWgZmHLxJ7JP0NI2V5HnyOyQkHydW1lkFPP'


output = replicate.run(
    "meta/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1",
    input={"prompt": "I am a language model"}
)

s = ''.join(output)
