import os
from http.server import HTTPServer

from REL.entity_disambiguation import EntityDisambiguation
from REL.ner import Cmns, load_flair_ner
from REL.server import make_handler

os.environ['CUDA_VISIBLE_DEVICES'] = ''

base_url = "/nfs/yding4/REL/data"

wiki_version = "wiki_2014"

config = {
    "mode": "eval",
    "model_path": "ed-wiki-2014",  # or alias, see also tutorial 7: custom models
}

model = EntityDisambiguation(base_url, wiki_version, config)

# Using Flair:
tagger_ner = load_flair_ner("ner-fast")

server_address = ("localhost", 5555)
server = HTTPServer(
    server_address,
    make_handler(
        base_url, wiki_version, model, tagger_ner
    ),
)

try:
    print("Ready for listening.")
    server.serve_forever()
except KeyboardInterrupt:
    exit(0)