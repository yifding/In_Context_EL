from bardapi import Bard

token = 'bghYx3NKgo2PrJL0TYxiDDdjoBfJKmdEmysIzuMMwNTXJbe7ipOZrkEv5ZvrRJKhtCLq1w.'
bard = Bard(token=token)
result = bard.get_answer("What is the current stock price of NVIDIA?")['content']
print(result)
