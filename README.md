# Document Retrieval via Context-Structured Keyword Augmentation and LLM-Based Question Generation

This are the implementation code from the techical paper "Document Retrieval via Context-Structured Keyword Augmentation and LLM-Based Question Generation".

Query_reformulation.py is the code file that implement the research proposed in the paper, with retrieval.py as the main file.

In this code, 5 datasets are tested, namely; Natural Question (NQ), Wizard of Wikipedia (WoW), ZeroshotRE, FEVER, and HotpotQA.

To generate the result, execute "python retrieval.py" creating files titled <tested_dataset>_proposed.json composing of the new query expanded. Furthermore, files titled <tested_dataset>_result.jsonl will be as the final result of retrieval using inner product similarity. To execute this script, please make sure to have downloaded the "metadata_flatip.index" and "flatip_400.db" files to support the retrieval process.
https://drive.google.com/drive/folders/1yDUCJ-5UUcKpqRvJVOazsHllnA4ww1pi?usp=sharing
