import numpy as np
import faiss
filename = 'test2dpoints.index'
load_file = False

query = np.expand_dims(np.array([-1,-1]), axis=0)
corpus = np.array([[0,0],
                    [1,2],
                    [2,1],
                    [-1,0],
                    [1,0],
                    [1,1],
                    [-1,-1]
                    ])

if not load_file:
    # Creamos faiss
    faiss_db = faiss.IndexFlatL2(corpus.shape[1])

    # Añadimos "embeddings" a faiss 
    faiss_db.add(corpus)

    # Guarda en disco el indice
    faiss.write_index(faiss_db, filename)
else:
    # Creamos faiss desde fichero
    faiss_db = faiss.read_index(filename)

D, I = faiss_db.search(query, k=3)

print(faiss_db.reconstruct(2))

print("D", D, "\nI", I)