import faiss
from sentence_transformers import SentenceTransformer
from tqdm.notebook import tqdm
import numpy as np
import sqlite3
import json
import time
import query_reformulation

if __name__ == "__main__":
    query_reformulation.main()
    
    data_names = ["fever", "hotpot", "nq", "wow", "zeroshotre"]
    
    embeddings = SentenceTransformer("thenlper/gte-small")
    index = faiss.read_index("flatip_400.index")
    
    top_k = 30

    for data in data_names:
        file_name = f"{data_names[i]}_proposed.json"
        time_array = []
        response_array = []
    
        # Open the database
        connection = sqlite3.connect('metadata_flatip.db')
    
        cursor = connection.cursor()
        
        # Open the JSON file and load its contents
        with open(file_name, 'r') as file:
            data = json.load(file)
        
            for item in tqdm(data, desc=f"Processing {data_names[i]}"):
                all_rankings = []
                
                query_id = int(item['id'])
                query_input = item['query']            
                query_keypoints = item['keypoints']
                query_question = item['generated_question']
        
                # For processing Question
                retrieved_array = []
                clean_array = []
        
                start_time = time.time()
                query_encoding = embeddings.encode(query_input).reshape(1, -1)
                distances, indices = index.search(query_encoding, top_k)
                    
                for retrieved_id in indices[0]:
                    cursor.execute('''
                        SELECT wikipedia_id FROM metadata WHERE hnsw_id = ?
                    ''', (int(retrieved_id),))
                
                    result = cursor.fetchone()
            
                    if result:
                        retrieved_array.append(result[0])
            
                # Make sure that each data_id are unique, avoid deduplication
                unique_array = list(set(retrieved_array))
                unique_array = list(dict.fromkeys(retrieved_array))
                
                all_rankings.append(unique_array)
                
        
                if len(all_rankings) > 1:
                    combined_test_run = rrf(all_rankings)
                
                elif len(all_rankings) > 0:
                    combined_test_run = all_rankings[0]
        
                end_time = time.time()
                time_array.append(end_time - start_time)
                
                test_data_input = {
                "id": query_id,
                "input": query_input,
                "output": [
                    {
                    "answer": "",
                    "provenance": [{"wikipedia_id": combined_test_run[0]},
                    	{"wikipedia_id": combined_test_run[1]},
                    	{"wikipedia_id": combined_test_run[2]},
                    	{"wikipedia_id": combined_test_run[3]},
                    	{"wikipedia_id": combined_test_run[4]}
                    ]
                    }
                ]
                }
                
                response_array.append(test_data_input)
        
                
        cursor.close()
        connection.close()
        
        # Dump the dictionary to a JSON file
        with open(f"{data_names[i]}_result.json", "w") as f:
            for item in response_array:
                f.write(json.dumps(item) + "\n")