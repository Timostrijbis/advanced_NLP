from datasets import load_dataset
import numpy as np
import json
# import cohere

def load_datasets(lang):
    ds1 = load_dataset("JRQi/Global-MMLU-emb", lang)
    # ds2 = load_dataset("Cohere/wikipedia-2023-11-embed-multilingual-v3", lang, split='train', streaming=True)
    ds2 = load_dataset("Cohere/wikipedia-2023-11-embed-multilingual-v3", "de", split='train')
    return ds1['test'], ds2

def retriever(questions, corpus):

    # co = cohere.Client("sr9NmsYnrQ2byCGENpTNy0m0mMyXgbcEJC7XXEYY")

    # Set number of retrieved passages
    top_k = 3

    # Load at max 1000 chunks + embeddings
    # max_docs = 100000
    # docs = []
    # doc_embeddings = []
    # for doc in corpus:
    #     docs.append(doc)
    #     doc_embeddings.append(doc['emb'])
    #     if len(docs) >= max_docs:
    #         break
    # doc_embeddings = np.asarray(doc_embeddings)
    doc_embeddings = np.asarray(corpus['emb'])

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
    lang = "es"
    questions, corpus = load_datasets(lang)


    output = retriever(questions, corpus)
    with open('data.json', 'w') as f:
      json.dump(output, f)


main()