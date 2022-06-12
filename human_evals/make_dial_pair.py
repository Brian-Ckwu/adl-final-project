import json
import random
from typing import List

def load_dials(jsonl: str) -> List[str]:
    dials = list()
    with open(jsonl) as f:
        for line in f:
            dial = json.loads(line.rstrip())["dialog"]
            dials.append(dial)
    return dials

def sample_dials(dials: List[str], indices: List[int]) -> List[str]:
    sampled_dials = list()
    for idx in indices:
        sampled = dials[idx]
        sampled_dials.append(sampled)
    return sampled_dials

if __name__ == "__main__":
    # load dialogues
    gen_3 = load_dials("../wlchen/num_generation-3.jsonl")
    gen_20 = load_dials("../wlchen/num_generation-20.jsonl")
    assert len(gen_3) == len(gen_20)

    # sample dialogues
    indices = random.sample(range(len(gen_3)), k=20)
    sampled_gen_3 = sample_dials(gen_3, indices)
    sampled_gen_20 = sample_dials(gen_20, indices)

    # anonymize and save
    to_save = [sampled_gen_3, sampled_gen_20]
    save_indices = random.sample(range(len(to_save)), k=len(to_save))
    for save_idx, obj in zip(save_indices, to_save):
        with open(file=f"{save_idx}.json", mode="wt") as f:
            json.dump(obj, f)
    
    # save mappings
    with open("./mappings.txt", mode="wt") as f:
        for save_idx in save_indices:
            f.write(str(save_idx) + '\n')