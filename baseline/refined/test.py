from refined.inference.processor import Refined
from refined.evaluation.evaluation import eval_all

refined = Refined.from_pretrained(
    model_name='wikipedia_model',
    entity_set="wikipedia",
    device="cuda:1",
)

results_numbers = eval_all(refined=refined, el=False)