from datasets import load_dataset
import numpy as np
import json
# import cohere

def load_datasets(lang):
    ds1 = load_dataset("JRQi/Global-MMLU-emb", lang)
    ds2 = load_dataset("Cohere/wikipedia-2023-11-embed-multilingual-v3", lang, split='train', streaming=True)
    return ds1['test'], ds2

def retriever(questions, corpus):

    # co = cohere.Client("sr9NmsYnrQ2byCGENpTNy0m0mMyXgbcEJC7XXEYY")

    # Set number of retrieved passages
    top_k = 3

    # Load at max 1000 chunks + embeddings
    max_docs = 5000000
    docs = []
    doc_embeddings = []
    for doc in corpus:
        docs.append(doc)
        doc_embeddings.append(doc['emb'])
        if len(docs) >= max_docs:
            break
    doc_embeddings = np.asarray(doc_embeddings)
    # embeddings = [x['emb'] for x in corpus]
    # doc_embeddings = np.asarray(embeddings)


    output_list = {}


    for i in range(len(questions)):
        question_id = (questions[i]['id'])
        print(question_id)
        query = questions[i]['question']
        embedding = [questions[i]['emb']]
        embedding = np.asarray(embedding)

        # Compute dot score between query embedding and document embeddings
        dot_scores = np.matmul(embedding, doc_embeddings.transpose())[0]
        top_k_hits = np.argpartition(dot_scores, -top_k)[-top_k:].tolist()

        # Sort top_k_hits by dot score
        top_k_hits.sort(key=lambda x: dot_scores[x], reverse=True)

        results = []
        for doc_id in top_k_hits:
            # print(docs[doc_id]['title'])
            # print(docs[doc_id]['text'])
            # print(docs[doc_id]['url'], "\n")
            results.append(docs[doc_id]['text'])
        output_list[question_id] = results
    return output_list


def main():

    # Set language
    lang = "ha"
    questions, corpus = load_datasets(lang)
    print(corpus)


    output = retriever(questions, corpus)
    with open(lang+'.json', 'w') as f:
      json.dump(output, f, indent=4)


main()