# QE Testing

This are the list of files that designed to test the performance of the proposed method. The main file is retrieval.py with query_reformulation.py as a file to generate the proposed context-full question form query. To run the test, execute "python retrieval.py" and it will take some time to executes. This will generate new files titled <tested_dataset>_proposed.json, that will be further retrieved in retrieval.py. After finishing the file execution, <tested_dataset>_proposed.json

To execute this script, please make sure to have downloaded the "metadata_flatip.db" and "flatip_400.db" files to support the retrieval process from the following link due to github file size constraint.
https://drive.google.com/drive/folders/1yDUCJ-5UUcKpqRvJVOazsHllnA4ww1pi?usp=sharing

Furthermore, this script also uses "gte-base" and "e5-base-v2" embeddings for encoding, "gte-small" embeddings for decoding, and "microsoft/Phi-3-mini-4k-instruct" which i run locally. Please expect these being downloaded into you local machine.

