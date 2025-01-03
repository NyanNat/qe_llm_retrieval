from datasets import load_dataset
import numpy as np
from datasets import concatenate_datasets
from keybert import KeyBERT
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
import spacy
import re
import json
from sentence_transformers import SentenceTransformer, util
import time
from tqdm import tqdm
import math

# --------> Load the LLM model for creative inference <--------
# Load the Phi-3-mini LLM model and its encoder
llm_model = AutoModelForCausalLM.from_pretrained( 
    "microsoft/Phi-3-mini-4k-instruct",  
    device_map="cuda",  
    torch_dtype="auto",  
    trust_remote_code=True,  
) 

# Encoding the input query to Phi-3-mini embeddings
llm_tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# The pipeline for llm to do its task and arguments to give limitations for llm generation
llm_pipe = pipeline( 
    "text-generation", 
    model=llm_model, 
    tokenizer=llm_tokenizer, 
) 

llm_generation_args = { 
    "max_new_tokens": 1024, 
    "return_full_text": False, 
    "temperature": 0.35,
    "top_p": 0.7,
    "do_sample": True,
}

# --------> Load embeddings and custom model embeddings declarations <--------
# Load the small-size gte-base mebeddings
tokenizer_gte = AutoTokenizer.from_pretrained("thenlper/gte-base")
model_gte = AutoModel.from_pretrained("thenlper/gte-base")

# Load the small-size e5-base embeddings
tokenizer_e5 = AutoTokenizer.from_pretrained("intfloat/e5-base-v2")
model_e5 = AutoModel.from_pretrained("intfloat/e5-base-v2")

# Load e5 on SentenceTransformer for similarity checking
similarity_embeddings_e5 = SentenceTransformer("intfloat/e5-base-v2")

# Custom embedding model that combine both thenlper/gte-base and intfloat/e5-base-v2 embeddings
class CustomEmbeddingModel:
    def __init__(self, tokenizer_gte, model_gte, tokenizer_e5, model_e5):
        self.tokenizer_gte = tokenizer_gte
        self.model_gte = model_gte
        self.tokenizer_e5 = tokenizer_e5
        self.model_e5 = model_e5
    
    def get_embedding(text, tokenizer, model):
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

    def combine_embeddings_concat(embedding1, embedding2):
        return np.concatenate((embedding1, embedding2))

    def extract_embeddings(self, texts):
        embeddings = []
        for text in texts:
            embedding_gte = self.get_embedding(text, tokenizer_gte, model_gte)
            embedding_e5 = self.get_embedding(text, tokenizer_e5, model_e5)
            combined_embedding = self.combine_embeddings_concat(embedding_gte, embedding_e5)
            embeddings.append(combined_embedding)
        return embeddings

custom_model = CustomEmbeddingModel(tokenizer_gte, model_gte, tokenizer_e5, model_e5)

# loading the KeyBERT using our custom embeddings for n-grams keypoints extraction
kw_model = KeyBERT(model=custom_model)

# --------> Code for NLP structure-based context extraction <--------
# Find all matches in the text according to "....." pattern
def semicolon_detection(doc):
    pattern = r'"(.*?)"'
    pattern_matches = re.findall(pattern, doc)

    return np.array(pattern_matches)

# To extract structure-based context using NOUN pattern
def noun_detection(doc, pos_tags):
    entity_array = []

    potential_entities = [(token.text, token.i) for token in doc if token.text[0].isupper()]
    
    # NOUN Pattern
    for npp in doc.noun_chunks:
        if npp.root.pos_ == 'NOUN' or npp.root.pos_ == 'PROPN':
            if npp.text.split()[0] == 'a' or npp.text.split()[0] == 'the':
                text_of_interest = npp.text.split()
                text_of_interest.pop(0)
                text_of_interest = ' '.join(text_of_interest)
                entity_array.append(text_of_interest)
            else:
                entity_array.append(npp.text)

    # NOUN Pattern
    index = 0
    while index < len(pos_tags):
        current_entity = pos_tags[index]
        word_tags = current_entity[1]

        if word_tags == 'NOUN' or word_tags == 'PROPN':
            entity_array.append(current_entity[0])
        
        index += 1
        
    # NOUN-NOUN Pattern
    index = 0
    while index < len(pos_tags):
        current_entity = pos_tags[index]
        word_tags = current_entity[1]

        # NOUN-NOUN Pattern
        if (word_tags == 'NOUN' or word_tags == 'PROPN') and index + 1 < len(pos_tags) and (pos_tags[index + 1][1] == 'NOUN' or pos_tags[index + 1][1] == 'PROPN'):
            next_entity = pos_tags[index + 1]
            combined_text = current_entity[0] + " " + next_entity[0]

            index += 2
            while index < len(pos_tags) and (pos_tags[index][1] == 'NOUN' or pos_tags[index][1] == 'PROPN'):
                combined_text += ' ' + pos_tags[index][0]
                index += 1

            entity_array.append(combined_text)

        else:
            index += 1
    

    index = 0
    while index < len(pos_tags):
        current_entity = pos_tags[index]
        word_tags = current_entity[1]

        # NOUN-ADP/DET-NOUN Pattern
        if (word_tags == 'NOUN' or word_tags == 'PROPN') and index + 2 < len(pos_tags) and (pos_tags[index + 1][1] == 'DET' or pos_tags[index + 1][1] == 'ADP') and (pos_tags[index + 2][1] == 'NOUN' or pos_tags[index + 2][1] == 'PROPN'):
            next_entity = pos_tags[index + 1]
            following_entity = pos_tags[index + 2]
            if (current_entity[0] in potential_entities) or (next_entity[0] in potential_entities) or (following_entity[0] in potential_entities):
                index += 1
                continue
            else:
                combined_text = current_entity[0] + " " + next_entity[0] + " " + following_entity[0]
    
                return_index = index - 1
                while return_index >= 0 and (pos_tags[return_index][1] == 'NOUN' or pos_tags[return_index][1] == 'PROPN'):
                    combined_text = pos_tags[return_index][0] + " " + combined_text
                    return_index -= 1
    
                index += 3
    
                while index < len(pos_tags) and (pos_tags[index][1] == 'NOUN' or pos_tags[index][1] == 'PROPN'):
                    combined_text += ' ' + pos_tags[index][0]
                    index += 1
    
                entity_array.append(combined_text)

        else:
            index += 1

    index = 0
    while index < len(pos_tags):
        current_entity = pos_tags[index]
        word_tags = current_entity[1]

        # NOUN-ADP-DET-NOUN Pattern
        if (word_tags == 'NOUN' or word_tags == 'PROPN') and index + 3 < len(pos_tags) and pos_tags[index + 1][1] == 'ADP' and pos_tags[index + 2][1] == 'DET' and (pos_tags[index + 3][1] == 'NOUN' or pos_tags[index + 3][1] == 'PROPN'):
            next_entity = pos_tags[index + 1]
            following_entity = pos_tags[index + 2]
            another_entity = pos_tags[index + 3]
            if (current_entity[0] in potential_entities) or (next_entity[0] in potential_entities) or (following_entity[0] in potential_entities) or (another_entity[0] in potential_entities):
                index += 1
                continue
            else:
                combined_text = current_entity[0] + " " + next_entity[0] + " " + following_entity[0] + " " + another_entity[0]

                return_index = index - 1
                while return_index >= 0 and (pos_tags[return_index][1] == 'NOUN' or pos_tags[return_index][1] == 'PROPN'):
                    combined_text = pos_tags[return_index][0] + " " + combined_text
                    return_index -= 1
    
                index += 4
    
                while index < len(pos_tags) and (pos_tags[index][1] == 'NOUN' or pos_tags[index][1] == 'PROPN'):
                    combined_text += ' ' + pos_tags[index][0]
                    index += 1
    
                entity_array.append(combined_text)

        else:
          index += 1

    return np.array(entity_array)

# Extract potential entities such as title or name by checking the capitalization on the first letter of the word
def uppercase_detection(doc):
    potential_entities = [(token.text, token.i) for token in doc if token.text[0].isupper()]

    entity_array = []

    index = 0
    while index < len(potential_entities):
        current_entity = potential_entities[index]
        word_indices = [current_entity[1]]
        entity_text = current_entity[0]

        while index + 1 < len(potential_entities) and potential_entities[index + 1][1] == current_entity[1] + 1:
            next_entity = potential_entities[index + 1]
            entity_text += " " + next_entity[0]
            current_entity = next_entity
            index += 1

        if len(entity_text.split()) > 1:
          entity_array.append(entity_text)
        index += 1

    return np.array(entity_array)

# To extract context using structure-based ADJ pattern
def adj_detection(pos_tags):
    entity_array = []

    index = 0
    while index < len(pos_tags):
        current_entity = pos_tags[index]
        word_tags = current_entity[1]

        # ADJ-NOUN
        if word_tags == 'ADJ' and index + 1 < len(pos_tags) and (pos_tags[index + 1][1] == 'NOUN' or pos_tags[index + 1][1] == 'PROPN'):
            combined_text = current_entity[0] + " " + pos_tags[index + 1][0]

            return_index = index - 1
            while return_index >= 0 and pos_tags[return_index][1] == 'ADJ':
              combined_text = pos_tags[return_index][0] + " "+ combined_text
              return_index -= 1

            index += 2

            while index < len(pos_tags) and (pos_tags[index][1] == 'NOUN' or pos_tags[index][1] == 'PROPN'):
                combined_text += ' ' + pos_tags[index][0]
                index += 1

            entity_array.append(combined_text)

        else:
            index += 1

    index = 0
    while index < len(pos_tags):
        current_entity = pos_tags[index]
        word_tags = current_entity[1]

        #ADJ-ADP/DET-NOUN
        if word_tags == 'ADJ' and index + 2 < len(pos_tags) and (pos_tags[index + 1][1] == 'DET' or pos_tags[index + 1][1] == 'ADP') and (pos_tags[index + 2][1] == 'NOUN' or pos_tags[index + 2][1] == 'PROPN'):
            next_entity = pos_tags[index + 1]
            following_entity = pos_tags[index + 2]
            combined_text = current_entity[0] + " " + next_entity[0] + " " + following_entity[0]


            return_index = index - 1
            while return_index >= 0 and pos_tags[return_index][1] == 'ADJ':
                combined_text = pos_tags[return_index][0] + " " + combined_text
                return_index -= 1

            index += 3

            while index < len(pos_tags) and (pos_tags[index][1] == 'NOUN' or pos_tags[index][1] == 'PROPN'):
                combined_text += ' ' + pos_tags[index][0]
                index += 1

            entity_array.append(combined_text)

        else:
            index += 1

    return np.array(entity_array)

# To extract context using structued-based VERB pattern
def verb_detection(pos_tags):
    entity_array = []

    index = 0
    while index < len(pos_tags):
        current_entity = pos_tags[index]
        word_tags = current_entity[1]

        # VERB-NOUN
        if word_tags == 'VERB' and index + 1 < len(pos_tags) and (pos_tags[index + 1][1] == 'NOUN' or pos_tags[index + 1][1] == 'PROPN'):
            combined_text = current_entity[0] + " " + pos_tags[index + 1][0]

            return_index = index - 1
            while return_index >= 0 and pos_tags[return_index][1] == 'VERB':
              combined_text = pos_tags[return_index][0] + " "+ combined_text
              return_index -= 1

            index += 2

            while index < len(pos_tags) and (pos_tags[index][1] == 'NOUN' or pos_tags[index][1] == 'PROPN'):
                combined_text += ' ' + pos_tags[index][0]
                index += 1

            entity_array.append(combined_text)

        else:
            index += 1

    index = 0
    while index < len(pos_tags):
        current_entity = pos_tags[index]
        word_tags = current_entity[1]

        # VERB-ADP/DET-NOUN
        if word_tags == 'VERB' and index + 2 < len(pos_tags) and (pos_tags[index + 1][1] == 'ADP' or pos_tags[index + 1][1] == 'DET') and (pos_tags[index + 2][1] == 'NOUN' or pos_tags[index + 2][1] == 'PROPN'):
            next_entity = pos_tags[index + 1]
            following_entity = pos_tags[index + 2]
            combined_text = current_entity[0] + " " + next_entity[0] + " " + following_entity[0]


            return_index = index - 1
            while return_index >= 0 and pos_tags[return_index][1] == 'VERB':
                combined_text = pos_tags[return_index][0] + " " + combined_text
                return_index -= 1

            index += 3

            while index < len(pos_tags) and (pos_tags[index][1] == 'NOUN' or pos_tags[index][1] == 'PROPN'):
                combined_text += ' ' + pos_tags[index][0]
                index += 1

            entity_array.append(combined_text)
            
        elif word_tags == 'VERB' and index + 1 < len(pos_tags) and pos_tags[index + 1][1] == 'ADP':
            next_entity = pos_tags[index + 1]
            combined_text = current_entity[0] + " " + next_entity[0]

            return_index = index - 1
            while return_index >= 0 and pos_tags[return_index][1] == 'VERB':
                combined_text = pos_tags[return_index][0] + " " + combined_text
                return_index -= 1
            
            index += 2
            entity_array.append(combined_text)
            
        else:
            index += 1

    index = 0
    while index < len(pos_tags):
        current_entity = pos_tags[index]
        word_tags = current_entity[1]

        # VERB-NOUN
        if word_tags == 'VERB' and index + 1 < len(pos_tags) and pos_tags[index + 1][1] == 'ADV':
            combined_text = current_entity[0] + " " + pos_tags[index + 1][0]

            return_index = index - 1
            while return_index >= 0 and pos_tags[return_index][1] == 'VERB':
              combined_text = pos_tags[return_index][0] + " "+ combined_text
              return_index -= 1

            index += 2

            while index < len(pos_tags) and pos_tags[index][1] == 'ADV':
                combined_text += ' ' + pos_tags[index][0]
                index += 1

            entity_array.append(combined_text)

        else:
            index += 1

    return np.array(entity_array)

# To extract context using structued-based ADVERB pattern
def adv_detection(pos_tags):
    entity_array = []

    index = 0
    while index < len(pos_tags):
        current_entity = pos_tags[index]
        word_tags = current_entity[1]

        # ADV-ADJ Pattern
        if word_tags == 'ADV' and index + 1 < len(pos_tags) and pos_tags[index + 1][1] == 'ADJ':
            next_entity = pos_tags[index + 1]
            combined_text = current_entity[0] + " " + next_entity[0]

            index += 2

            entity_array.append(combined_text)

        else:
            index += 1

    return np.array(entity_array)

# To extract context using SpaCy entity detection
def entity_detection(doc):
    entities = [ent.text for ent in doc.ents]
    return np.array(entities)

# To remove duplicates that are present in the final array, if a string is already inside another string
def compress_duplicates(entity_array):
    
    # Sort array by length of strings (longest first)
    sorted_array = sorted(entity_array, key=len, reverse=True)

    compressed_array = []

    for i, entity in enumerate(sorted_array):

        # Check if the current entity is a substring of any already added entity
        if not any(entity in e for e in compressed_array):
            compressed_array.append(entity)

    return np.array(compressed_array)


def pattern_context_extraction(texting):
    # Load SpaCy model
    nlp = spacy.load("en_core_web_sm")

    # Check the sentence for item that is closed by semicolon ("...")
    pattern_matches = semicolon_detection(texting)

    # processing text using spacy embedings
    processed_text = nlp(texting)

    # Creating an array containing the text itself and its Part-of-Speech
    pos_tags = [(token.text, token.pos_) for token in processed_text]

    # Check and extract for item that started with a splhabaet capitalization
    uppercase_array = uppercase_detection(processed_text)

    # Extract word-based tagged entity
    entity_array = entity_detection(processed_text)

    # Extract adjective-based context
    adj_array = adj_detection(pos_tags)

    # Extract noun-based context
    noun_array = noun_detection(processed_text, pos_tags)

    # Extract verb-based context
    verb_array = verb_detection(pos_tags)

    # Extract adverb-based context
    adv_array = adv_detection(pos_tags)

    # Combine all the previous extracted array and remove any duplicates
    combined_array = np.union1d(adv_array, np.union1d(entity_array, np.union1d(np.union1d(np.union1d(pattern_matches, uppercase_array), adj_array), np.union1d(noun_array, verb_array))))
    reduced_combination = compress_duplicates(combined_array)

    return reduced_combination

def generate_question(keyword):

    prompt = f'''You will be given a factual statement.
    First, generate a simple, direct, short and straighforward question to seek information not provided within the keypoints itself.
    Then, generate a more creative and innovative question that encourages exploration of the keypoints without.
    Only use context and entities present in the keypoint and do not add any context or entities outside the ones present in keypoints.
    keypoint: {keyword}
    1. Simple Question:
    2. Creative Question:'''
        

    # Message for instruct LLM to invoke creativity
    messages = [
        {"role": "system", "content": "You are a factual question generation AI assistant."},
        {"role": "user", "content": prompt}
    ]
    
    # Generate output text from LLM
    output = llm_pipe(messages, **llm_generation_args)
    output_sentence = output[0]['generated_text']

    return output_sentence

def main():
    # Load the dataset of KILT dataset 
    ds_fever = load_dataset("facebook/kilt_tasks", "fever", split="test")
    ds_hotpot = load_dataset("facebook/kilt_tasks", "hotpotqa", split="test")
    ds_nq = load_dataset("facebook/kilt_tasks", "nq", split="test")
    ds_wow = load_dataset("facebook/kilt_tasks", "wow", split="test")
    ds_zeroshotre = load_dataset("facebook/kilt_tasks", "structured_zeroshot", split="test")
    
    # Put the data of interest and its corresponding names in a list
    data_list = [ds_fever, ds_hotpot, ds_nq, ds_wow, ds_zeroshotre]
    data_names = ["fever", "hotpot", "nq", "wow", "zeroshotre"]
    
    for i, dataset in enumerate(data_list):
        # To store all extracted testing
        all_array = []
    
        # To store all time aspect of testing
        overall_time = []
        
        for index, question in tqdm(enumerate(dataset), total=len(dataset), desc='Processing'+ data_names[i]):
            # Get the sentences that we wanted to extract
            text_id = question['id']
            text = question['input']
    
            # Starting the timer for time test
            start_time = time.time()
                
            original_text_data = text.replace('[SEP]', '')
                
            # Executing NLP-structure-based context extraction
            context_array = pattern_context_extraction(original_text_data)
                
            # To decide the number of keypoint based on length of query and restrict to be between 1 and 5
            context_len = min(5, math.ceil(len(text.split())/12))
        
            # Executing the keypoint extraction using KeyBERT
            keywords = kw_model.extract_keywords(original_text_data, keyphrase_ngram_range=(2, 8), use_maxsum=True, nr_candidates=25, stop_words='english', top_n=context_len)
            final_keywords = [item[0] for item in keywords]
                
            # To store words that hasn't been used during expansion
            unused_keywords = []
            # To check word-based similarities between structure-based context and n-gram based keypoint
            for origin_index, context_keyword in enumerate(context_array):
                    # Array that contain the words of structure-based context
                    words1 = np.array(context_keyword.lower().split())
        
                    # variable to denote of how many times the keyword has been used and the array to store the unused keyword
                    origin_usage_num = 0
        
                    for index, final_keyword in enumerate(final_keywords):
                            # Array that contains the words of extracted n-grams keypoints
                            words2 =  np.array(final_keyword.lower().split())
        
                            # To get the common word between the two array
                            common_keywords = np.intersect1d(words1, words2)
        
                            # To check if the number of common words between structure-based context extraction and n-gram based keypoint extractions
                            # If the length of common words and context_keywords are the same, meaning all the context are in the keypoints, we don't need to do anything
                            # if the length of common words is shorter, meaning we will continue the process
                            if len(common_keywords) == len(words1):
                                    # To denote that the extracted context has been utilized
                                    origin_usage_num += 1
                                        
                            # If the common words packed more than 70% of the keypoint length, we just replace the extracted keypoints since it is likely that both have the same context
                            elif len(common_keywords) >= 0.7*len(words2):
        
                                    # To denote that the extracted context has been utilized
                                    origin_usage_num += 1
                                    final_keywords[index] = context_keyword
        
                            # If the common words packed are below 70% of the keypoint length and not zero, we will proceed with the extraction
                            elif len(common_keywords) > 0:
        
                                    # To denote that the extracted context has been utilized
                                    origin_usage_num += 1
        
                                    # Get the index of the common words in the extracted keypoint
                                    common_index = [np.where(words2 == common)[0][0] for common in common_keywords]
                                    # Pick the common words with the lowest index
                                    common_index.sort()
                                    lowest_common_index = common_index.pop(0)
                                                
                                    # Remove other common words between the two aside from the most-right index from the extracted keypoints
                                    for com_index in sorted(common_index, reverse=True):
                                            words2 = np.delete(words2, com_index)
        
                                    # Create the combined phrase
                                    first_common = words2[lowest_common_index]
                                    before_common = ' '.join(words2[:lowest_common_index])
                                    after_common = ' '.join(words2[lowest_common_index + 1:])
                                    final_keywords[index] = f"{before_common} {context_keyword} {after_common}"
                        
                    # After iterating through all extracted n-gram keypoints, check if there is any context that hasn't been utilized yet
                    if origin_usage_num == 0:
                            unused_keywords.append(context_keyword)
                
        
            # If there is any unused keywords, we append the keywords to add more context using embeddings similarities
            if len(unused_keywords) > 0:
                    for unused in unused_keywords:
        
                            # Create embeddings for the unused keyword(s)
                            unused_word_embeddings = similarity_embeddings_e5.encode(unused)
        
        
                            for unused_index, final_keyword in enumerate(final_keywords):
                                    first_word, last_word = final_keyword.split()[0], final_keyword.split()[-1]
        
                                    first_embeddings = similarity_embeddings_e5.encode(first_word)
                                    last_embeddings = similarity_embeddings_e5.encode(last_word)
                                        
                                    first_similarity = util.cos_sim(unused_word_embeddings, first_embeddings).item()
                                    last_similarity = util.cos_sim(unused_word_embeddings, last_embeddings).item()
        
                                    if first_similarity >= last_similarity and first_similarity > 0.7:
                                            final_keywords[unused_index] = f'{unused} {final_keyword}'
        
                                    elif last_similarity >= first_similarity and last_similarity > 0.7:
                                            final_keywords[unused_index] = f'{final_keyword} {unused}'
        
            # Array to store clean question for retrieval
            processed_lines = []
    
            # Proceeds to generate question as query
            for keyword in final_keywords:
                
                final_question= generate_question(keyword)
        
                # Split the outputted text into lines
                lines = final_question.splitlines()
        
                # Get the clean version of each question
                clean_lines = [line.lstrip() for line in lines if len(line.split()) > 3]
        
                # Clean the preeceding question explanation
                for line in clean_lines:
                    if line[0] == '1' or line[0] == '2':
                        processed_lines.append(' '.join(line.split()[3 : len(line.split())]))
                    elif line[0] == '-':
                        processed_lines.append(' '.join(line.split()[1 : len(line.split())]))
                    else:
                        processed_lines.append(line)
        
            # Measure the time of the end
            end_timeout = time.time()
        
            # Add the new entry for all dataset
            new_entry = {
                "id": text_id,
                "query": text,
                "keypoints": final_keywords,
                "generated_question": processed_lines
            }
        
            all_array.append(new_entry)
            torch.cuda.empty_cache()
        
            overall_time.append(end_timeout - start_time)
    
        overall_time_mean = np.mean(np.array(overall_time))
        
        # Save to JSON file
        file_name = f"{data_names[i]}_proposed.json"
        with open(file_name, 'w') as json_file:
            json.dump(all_array, json_file, indent=4)
    
        print(f"Saved {file_name} with {len(all_array)} entries.")