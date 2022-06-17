import argparse
import json
import random
from collections import defaultdict
from typing import List, Dict
import math

import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BlenderbotForConditionalGeneration,
    BlenderbotTokenizer
)
import spacy

from sentence_transformers import SentenceTransformer, util

# Fix random seed for reproducibility
def set_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# get lemmatized keyword, tjis is only used in check_hit() 
def get_lemma_keywords(nlp):

    with open("keywords.json") as f:
        keywords = json.load(f)

    # lemmatize words in keywords
    for key, val in keywords.items():
        # separate words by its length (one, others)
        one_lemma = []
        multi_lemma = []
        for word in val:
            split = [token.lemma_ for token in nlp(word)]
            if len(split) >= 2:
                multi_lemma.append(" ".join(split))
            else:
                one_lemma.append(split[0])
            keywords[key] = [one_lemma, multi_lemma]
    
    return keywords

# check whether the utterance has hit the lemma_keyword
def check_hit(utterance, lemma_keywords, nlp):
    
    lemma_utterance = [token.lemma_ for token in nlp(utterance)]
    service_hits = defaultdict(int)
    for key, (one, multi) in lemma_keywords.items():
        intersection = set(one) & set(lemma_utterance)
        # check whether the word, the length is bigger than 2, is in the utterance
        for m in multi:
            unsplit_utterance = " ".join(lemma_utterance)
            if m in unsplit_utterance:
                intersection.add(m)
        service_hits[key] += len(intersection)
        
    # Is there a keyword in this utterance
    isService = sum(service_hits.values()) != 0

    return True if isService else False

# construct the topic corpus from keywords for semantic search
def construct_topic_corpus():
    with open("keywords.json") as f:
        keywords = json.load(f)

    topic_corpus = []
    for k, v in keywords.items():
        topic_corpus.append(
            " ".join(v)
        )
    
    return topic_corpus

# construct the persona corpus from bst dataset for semantic search
def construct_persona_corpus(split='train'):
    bst_personas = load_dataset("blended_skill_talk", split=split)['personas']
    persona_corpus = [sent for persona in bst_personas for sent in persona]

    return persona_corpus

# do semantic search, return the ids of the top_k scoring elements in corpus
def semantic_search(query: str, corpus: List[str], top_k: int, embedder) -> List[int]:

    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    query_embedding = embedder.encode([query], convert_to_tensor=True)
    
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)[0]
    hit_ids = [hit['corpus_id'] for hit in hits]

    return hit_ids


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default="facebook/blenderbot-400M-distill",
        type=str,
        help="model to chat with simulator",
    )

    parser.add_argument("--num_chats", default=980, type=int, help="the number of round")

    parser.add_argument("--split", default="test", type=str, help="split")

    parser.add_argument("--seed", default=26, type=int, help="random seed")

    parser.add_argument(
        "--interactive_mode",
        action="store_true",
        help="make the simualtor interact with the user (type 'stop' to stop the program, type 'exit' to leave current round)",
    )

    parser.add_argument(
        "--output",
        default="output.jsonl",
        type=str,
        help="file to save the dialogs",
    )

    parser.add_argument(
        "--disable_output_dialog",
        action="store_true",
        help="whether output the dialogs to the command line",
    )

    ###### custom args ######
    parser.add_argument(
        "--num_generation",
        default=3,
        type=int,
        help="maximum number of generation allowed to try in each turn for bot",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
    )
    #########################
    
    args = parser.parse_args()

    return args


def preprocess(example):

    example["personas"] = [f"your persona: {p}" for p in example["personas"]]
    example["context"] = "\n".join(
        example["personas"]
        + (["additional_context"] if example["additional_context"] else [])
        + example["previous_utterance"]
    )

    return example



if __name__ == "__main__":
    args = parse_args()

    set_seeds(args.seed)
    # random.seed(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"

    mname = args.model_name_or_path
    simulator = BlenderbotForConditionalGeneration.from_pretrained(mname).to(device)
    simulator_tokenizer = BlenderbotTokenizer.from_pretrained(mname)

    # load your bot
    bot = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path).to(device)
    bot_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    dataset = load_dataset("blended_skill_talk", split=args.split)
    dataset = dataset.map(
        preprocess,
        remove_columns=[
            "free_messages",
            "guided_messages",
            "suggestions",
            "personas",
            "additional_context",
            "previous_utterance",
        ],
    )

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    lemma_keywords = get_lemma_keywords(nlp=nlp)

    # embedder = SentenceTransformer('msmarco-distilbert-base-v4')
    embedder = SentenceTransformer('all-MiniLM-L6-v2') 
    topic_corpus = construct_topic_corpus()
    persona_corpus = construct_persona_corpus()

    topic_id_map = {
        0: 'restaurant',
        1: 'hotel',
        2: 'movie',
        3: 'song',
        4: 'transportation',
        5: 'attraction'
    }


    if args.interactive_mode:
        raise NotImplementedError
    else:
        assert args.num_chats <= len(
            dataset
        ), f"--num_chats needs to be smaller than dataset (<={len(dataset)})"
        dataset = dataset.select(random.sample(range(len(dataset)), args.num_chats))

        output = []
        for index, context in enumerate(
            tqdm(dataset["context"], disable=(not args.disable_output_dialog))
        ):  
            dialog = []
            
            bot_decode_args = {
                'do_sample': True,
                'top_k': 120,
                'top_p': 0.8,
                'no_repeat_ngram_size': 4,
                'repetition_penalty': 1.5,
            }
            
            if not args.disable_output_dialog:
                print(f" dialog id: {index}")
            for turn in range(6):
        
                inputs = simulator_tokenizer(
                    [
                        "</s> <s>".join(
                            ([context] + dialog if len(dialog) < 3 else dialog[-3:])
                        )
                    ],
                    return_tensors="pt",
                    truncation=True,
                ).to(device)

                reply_ids = simulator.generate(**inputs)
                
                text = simulator_tokenizer.batch_decode(
                    reply_ids, skip_special_tokens=True
                )[0].strip()
                
                dialog.append(text)

                if (turn != 0 and check_hit(utterance=text, lemma_keywords=lemma_keywords, nlp=nlp)) or (turn == 5):
                    break


                if not args.disable_output_dialog:
                    print(f"\033[0;32;49m {'simulator: ': ^11}{text} \033[0;0m")


                # start Semantic-Search Chain (SSC) to build bot persona
                
                # Chian-1
                # find relevent topic, with dialouge history as query
                query = " ".join(dialog)
                corpus_ids = semantic_search(
                    query=query,
                    corpus=topic_corpus,
                    top_k=1,
                    embedder=embedder
                )
                                
                # Chian-2
                # find top_k relevent keywords from selected topic from chain-1, with dialouge history as query
                with open("keywords.json") as f:
                    keywords = json.load(f)
                candidate_keywords_corpus = keywords[topic_id_map[corpus_ids[0]]]
                corpus_ids = semantic_search(
                    query=query,
                    corpus=candidate_keywords_corpus,
                    top_k=5,
                    embedder=embedder
                )

                # Chian-3
                # find top_k persona sentences from BST trainset personas, with keywords find by chain-2 as query
                query = " ".join([candidate_keywords_corpus[id] for id in corpus_ids])
                corpus_ids = semantic_search(
                    query=query,
                    corpus=persona_corpus,
                    top_k=2,
                    embedder=embedder
                )
                    

                ###### Build bot persona ######                   
                bot_persona = "\n".join(
                    [
                        f"your persona: {persona_corpus[id]}" for id in corpus_ids
                    ]
                )
                ###############################

                inputs = bot_tokenizer(
                    [
                        "</s> <s>".join(
                            ([bot_persona] + dialog[-3:])
                        )
                    ],
                    return_tensors="pt",
                    truncation=True
                ).to(device)

                for _ in range(0, args.num_generation):
                    reply_ids = bot.generate(
                        **inputs,
                        **bot_decode_args
                    )
                            
                    text = bot_tokenizer.batch_decode(
                        reply_ids,
                        skip_special_tokens=True
                    )[0].strip()

                    if check_hit(utterance=text, lemma_keywords=lemma_keywords, nlp=nlp):
                        break

                dialog.append(text)

                if not args.disable_output_dialog:
                    print(f"\033[0;33;49m {'bot: ': ^11}{text} \033[0;0m")

            output.append(dialog)

            if not args.disable_output_dialog:
                print()

        with open(args.output, "w") as f:
            for idx, dialog in enumerate(output):
                f.write(json.dumps({"id": idx, "dialog": dialog}) + "\n")
