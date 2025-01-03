import faiss
from sentence_transformers import SentenceTransformer
from tqdm.notebook import tqdm
import numpy as np
import sqlite3
import json
import time
from ranx import fuse, Run
import query_reformulation

if __name__ == "__main__":
    query_reformulation.main()
    
    data_names = ["fever", "hotpot", "nq", "wow", "zeroshotre"]
    
    embeddings = SentenceTransformer("thenlper/gte-small")
    index = faiss.read_index("flatip_400.index")

    top_k = 25
    
    time_array = []
    response_array = []
    
    for i, data in enumerate(data_names):
        file_name = f"{data}_proposed.json"
        # Open the database
        connection = sqlite3.connect('metadata_flatip.db')
        
        cursor = connection.cursor()
        
        # Open the JSON file and load its contents
        with open(file_name, 'r') as file:
            data = json.load(file)
        
            for item in tqdm(data, desc=f"Processing {data} Question Query"):
                all_rankings = []
                try:
                    query_id = int(item['id'])
                except:
                    query_id = item['id']
                query_input = item['query']            
                query_keypoints = item['keypoints']
                query_question = item['generated_question']
        
                query_input = [query_input]
                query_all = query_input + query_question + query_keypoints
        
                start_time = time.time()
                
                # For processing Question
                for key in query_all:
                    
                    retrieved_array = []
                    clean_array = []
                    
                    query_encoding = embeddings.encode(key).reshape(1, -1)
                    distances, indices = index.search(query_encoding, top_k)
        
                    for retrieved_id in indices[0]:
                        cursor.execute('''
                            SELECT wikipedia_id FROM metadata WHERE hnsw_id = ?
                        ''', (int(retrieved_id),))
                
                        result = cursor.fetchone()
            
                        if result:
                            retrieved_array.append(result[0])
                            
                    # Make sure that each data_id are unique, avoid deduplication
                    unique_array = list(dict.fromkeys(retrieved_array))
        
                        
                    first_indices = [retrieved_array.index(item) for item in unique_array]
                    first_distances = [distances[0][ind] for ind in first_indices]
                    negated_distances = [-dist for dist in first_distances]
                    all_array = Run.from_dict({"q1": dict(zip(unique_array, negated_distances))})
                        
                    all_rankings.append(all_array)
        
                if len(all_rankings) > 1:
                    combined_test_run = fuse(runs = all_rankings, method='rrf', params={"k": 60})
                    final_run = list(Run.to_dict(combined_test_run)['q1'].items())
                        
                elif len(all_rankings) == 1:
                    combined_test_run = all_rankings[0]
                    final_run = list(Run.to_dict(combined_test_run)['q1'].items())
        
                end_time = time.time()
                time_array.append(end_time - start_time)
                
                test_data_input = {
                "id": query_id,
                "input": query_input,
                "output": [{
                    "answer": "",
                    "provenance": [
                        {"wikipedia_id": final_run[0][0]},
                        {"wikipedia_id": final_run[1][0]},
                        {"wikipedia_id": final_run[2][0]},
                        {"wikipedia_id": final_run[3][0]},
                        {"wikipedia_id": final_run[4][0]},
                    ]
                    }
                ]
                }
                        
                response_array.append(test_data_input)
        
        cursor.close()
        connection.close()
        
        # Dump the dictionary to a JSON file
        with open(f"{data}_result.jsonl", "w") as f:
            for item in response_array:
                f.write(json.dumps(item) + "\n")
