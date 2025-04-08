# advanced_NLP
Final project for the course Advanced NLP

USAGE:

1. Install required dependencies:

*pip install datasets numpy torch transformers

2. Login to your huggingface account

*huggingface-cli login
You need to generate a (read) access token, and you need access to the LLAMA 3 gated repository. You can request access here: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

3. Run the retriever.py

*python retriever.py
Current language is currently set to Hausa, if you want to change the language, you can do so in line 62.
This script will output the retrieved chunks to a JSON file.
WARNING: Choosing a language with a large number of wiipedia documents will require a lot of RAM.

4. Run the generator

*python llama.py
If you have changed the language from Hausa to another language, you need to change the name of the file from ha.json to the correct file name in line 163.  
Furthermore, an adjustment to the prompt is needed, line 79. Finally, you need to change the test dataset to your desired language abbreviations (e.g "el" for Greek)
