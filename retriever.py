from datasets import load_dataset
import numpy as np
import json

def load_datasets(lang):
    ds1 = load_dataset("JRQi/Global-MMLU-emb", lang)
    ds2 = load_dataset("Cohere/wikipedia-2023-11-embed-multilingual-v3", lang, split='train', streaming=True) # Load the dataset in streaming mode
    return ds1['test'], ds2

def retriever(questions, corpus):

    # Set number of retrieved passages
    top_k = 3

    # If you're running a language with too much data, uncomment the next line and also lines 24 and 25
    max_docs = 5000000


    docs = []
    doc_embeddings = []
    for doc in corpus:
        docs.append(doc)
        doc_embeddings.append(doc['emb'])
        # if len(docs) >= max_docs:
        #     break
    doc_embeddings = np.asarray(doc_embeddings)

    output_list = {}


    # Iterate through each question to get the embeddings
    for i in range(len(questions)):
        question_id = (questions[i]['id'])
        print(question_id)
        query = questions[i]['question']
        embedding = [questions[i]['emb']]
        # Turn the embedding into a numpy array
        embedding = np.asarray(embedding)

        # Compute dot score between query embedding and document embeddings
        dot_scores = np.matmul(embedding, doc_embeddings.transpose())[0]
        # Sort the dot scores in descending order, only select the top-k passages
        top_k_hits = np.argpartition(dot_scores, -top_k)[-top_k:].tolist()

        # Sort top_k_hits by dot score
        top_k_hits.sort(key=lambda x: dot_scores[x], reverse=True)

        results = []
        for doc_id in top_k_hits:
            results.append(docs[doc_id]['text'])
        output_list[question_id] = results
    return output_list


def main():

    # Set language
    lang = "ha"
    questions, corpus = load_datasets(lang)

    output = retriever(questions, corpus)

    # Save the output to a JSON file
    with open(lang+'.json', 'w') as f:
      json.dump(output, f, indent=4)


main()