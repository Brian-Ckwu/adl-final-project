import argparse
import json
import random
from collections import defaultdict
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BlenderbotForConditionalGeneration,
    BlenderbotTokenizer,
)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PhrasalConstraint
import spacy
from sentence_transformers import SentenceTransformer, util
import random
import numpy as np

topic_id_map = {
    0: 'restaurant',
    1: 'hotel',
    2: 'movie',
    3: 'song',
    4: 'transportation',
    5: 'attraction'
}

with open("keywords_in_dataset.json") as f:
    keywords_in_dataset = json.load(f)

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

# construct the corpus from keywords for semantic search
def construct_keyword_corpus():
    with open("keywords.json") as f:
        keywords = json.load(f)

    keyword_corpus = []
    for k, v in keywords.items():
        keyword_corpus.append(
            " ".join(v)
        )
    
    return keyword_corpus

# do semantic search, return the id of the highest scoring topic
def get_topic(query, corpus, embedder) -> int:

    # Topic Level
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    query_embedding = embedder.encode([query], convert_to_tensor=True)
    
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)[0] 
    top_hit = hits[0]['corpus_id']
    top_hit_score = [hits[i]['score'] for i in range(len(hits))]
 
    #print('Topic score:')
    #for hit in hits: 
        #print(topic_id_map[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']), end = ', ')
    #print()

    # Token Level
    keyword_embeddings = embedder.encode(corpus[top_hit].split(), convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, keyword_embeddings, top_k=5)[0] 
    top_keyword_list = [corpus[top_hit].split()[hits[i]['corpus_id']]  for i in range(len(hits))]
    
    weights = [hits[i]['score'] for i in range(len(hits))]
    #weights_accumulated = np.cumsum(weights)


    top_keyword_list = random.choices(population = top_keyword_list, weights = weights, k = 5)
    top_keyword_score = [hits[i]['score'] for i in range(len(hits))]

    #print('Keyword score:')
    #for hit in hits: 
        #print(corpus[top_hit].split()[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']), end = ', ')
    #print()

    # Direct Token Level
    '''
    all_keyword = ''
    for i in range(len(corpus)):
        all_keyword += corpus[i]
    all_keyword_list = list(dict.fromkeys(all_keyword.split()))
    
    Direct_keyword_embeddings = embedder.encode(all_keyword_list, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, Direct_keyword_embeddings, top_k=5)[0] 
    direct_top_keyword_list = [all_keyword_list[hits[i]['corpus_id']]  for i in range(len(hits))]
    direct_top_keyword_list = random.choices(population = direct_top_keyword_list, weights = [hits[i]['score'] for i in range(len(hits))], k = 3)

    print('Direct Keyword score:')
    for hit in hits: 
        print(all_keyword_list[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']), end = ', ')
    print()
    '''

    return top_hit, top_hit_score, top_keyword_list, top_keyword_score

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default="facebook/blenderbot-400M-distill",
        type=str,
        help="model to chat with simulator",
    )

    parser.add_argument("--num_chats", default=5, type=int, help="the number of round")

    parser.add_argument("--split", default="train", type=str, help="split")

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
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mname = "facebook/blenderbot-400M-distill"
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

    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    #embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    #embedder = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L6-cos-v5')

    keyword_corpus = construct_keyword_corpus()

    if args.interactive_mode:   
        for _ in range(args.num_chats):
            dialog = ["hi"]
            while True:
                inputs = simulator_tokenizer(
                    ["</s> <s>".join(dialog[-3:])], return_tensors="pt", truncation=True
                ).to(device)
                reply_ids = simulator.generate(**inputs, do_sample=True, top_p=0.8)
                text = simulator_tokenizer.batch_decode(
                    reply_ids, skip_special_tokens=True
                )[0].strip()
                dialog.append(text)
                print(f"\033[0;32;49m {'simulator: ': ^11}{text} \033[0;0m")

                text = input(f"\033[0;33;49m {'you: ': ^11}")
                dialog.append(text)
                if text in ["stop", "exit"]:
                    break
            if text == "stop":
                break
            print()
    else:
        assert args.num_chats <= len(
            dataset
        ), f"--num_chats needs to be smaller than dataset (<={len(dataset)})"
        dataset = dataset.select(random.sample(range(len(dataset)), args.num_chats))

        output = []
        transition_success = 0
        for index, context in enumerate( 
            tqdm(dataset["context"], disable=(not args.disable_output_dialog))
        ):
            dialog = []
            if not args.disable_output_dialog:
                print(f" dialog id: {index}")
 
            use_transition = False
            already_success = False

            print('context = \n{}'.format(context))

            for turn in range(5):
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

                if not args.disable_output_dialog:
                    print(f"\033[0;32;49m {'simulator: ': ^11}{text} \033[0;0m")

                if turn != 0 and check_hit(utterance=text, lemma_keywords=lemma_keywords, nlp=nlp): 
                    if use_transition and not already_success:
                        transition_success += 1 
                        print(f"\033[0;34;49m {'Success!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!': ^11} \033[0;0m") 
                    
                    already_success = True
                    #break

                constrained_words = ''
                bot_persona = ''

                if turn >= 0 and turn < 4: 
                    #print('=================================== Analyzing Semantics ===================================')
                    use_transition = True

                    query = " ".join(dialog)
                    topic_id, top_hit_score, top_keyword, top_keyword_score = get_topic(
                        query=query,
                        corpus=keyword_corpus,
                        embedder=embedder
                    )
                     
                    threshold = -100000.0

                    top_keyword_persona = []
                    for keyword in top_keyword:
                        if not (keyword in keywords_in_dataset.keys()):
                            top_keyword_persona.append('I like ' + keyword + '.')
                            continue
                        if len(keywords_in_dataset[keyword]) > 0:
                            #print(keywords_in_dataset[keyword])
                            #top_keyword_persona.append(random.choices(population = keywords_in_dataset[keyword], k = 1)[0])
 
                            corpus_embeddings = embedder.encode(keywords_in_dataset[keyword], convert_to_tensor=True)
                            query_embedding = embedder.encode([query], convert_to_tensor=True) 
                            hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)[0]  

                            weights = [hits[i]['score'] for i in range(len(hits))]
                            tmp_list = [keywords_in_dataset[keyword][hits[i]['corpus_id']]  for i in range(len(hits))]
                            choice = random.choices(population = tmp_list, weights = weights, k = 1)  
                            top_keyword_persona.append(choice[0])
                             
                        else:
                            top_keyword_persona.append('I like ' + keyword + '.') 

                    top_keyword = top_keyword_persona
 
                    ### Build keyword constraint ### 
                    #if (top_hit_score[0] > threshold and top_keyword_score[0] > threshold) and not already_success:
                        #constrained_words = top_keyword[0]
                    #print('Selected word: {}'.format(constrained_words))
                    ############################################################################################################################

                    ###### Build bot persona ######
                    #random_keyword = random.sample(keywords[topic_id_map[topic_id]], 3) 
                    #bot_persona = f"your persona: I like {random_keyword[0]}, {random_keyword[1]} and {random_keyword[2]}" 

                    #print('top_hit_score = {}'.format(top_hit_score))
                    #print('top_keyword_score = {}'.format(top_keyword_score)) 
 
                    #if (top_hit_score[0] > threshold and top_keyword_score[0] > threshold) and (not already_success): 
                        #bot_persona = f"your persona: I like {top_keyword[0]}, {top_keyword[1]} and {top_keyword[2]}" 
                    #bot_persona = f"your persona: I like {direct_top_keyword_list[0]}, {direct_top_keyword_list[1]} and {direct_top_keyword_list[2]}"  
                    
                    #print(top_keyword)
                    bot_persona = " ".join(
                            (['your persona:'] + top_keyword[0:2] + ['\nyour persona:'] + top_keyword[2:4])
                        )

                    print(bot_persona)
                    ############################################################################################################################

                    #print('============================= Analyzing Semantics Completed =============================')



                # you might need to change this line due to the model you use 
                inputs = bot_tokenizer(
                    [
                        "</s> <s>".join(
                            ([bot_persona] + dialog[-3:] if turn >= 2 else dialog[-3:])
                        )
                    ],
                    return_tensors="pt",
                    truncation=True
                ).to(device)

 
                if constrained_words != '':
                    constraints = [
                        PhrasalConstraint(
                            bot_tokenizer(constrained_words, add_special_tokens=False).input_ids
                        )
                    ]
                    reply_ids = bot.generate(**inputs, constraints=constraints,    num_beams=30,    num_return_sequences=1,    no_repeat_ngram_size=1,    remove_invalid_values=True)
                else:
                    #reply_ids = bot.generate(**inputs)
                    reply_ids = bot.generate(**inputs, do_sample=True, top_p=0.8, temperature = 1.5)
                
 
                text = bot_tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[
                    0
                ].strip()
                dialog.append(text)
                if not args.disable_output_dialog:
                    print(f"\033[0;33;49m {'bot: ': ^11}{text} \033[0;0m")
 
            output.append(dialog)
            if not args.disable_output_dialog:
                print()

        print('Transition Success Rate: {} / {}'.format(transition_success, args.num_chats))

        with open(args.output, "w") as f:
            for idx, dialog in enumerate(output):
                f.write(json.dumps({"id": idx, "dialog": dialog}) + "\n")
