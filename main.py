"""
Lab 4
Semantic Data Ingestion & Chunking Optimization

Aim:
1. Load PDF and HTML documents.
2. Extract clean text.
3. Perform Recursive Character Splitting.
4. Perform Semantic Chunking.
5. Compare both chunking strategies.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --------------------------------------------------
# Step 1 : Load PDF
# --------------------------------------------------

print("Loading PDF...")

pdf_loader = PyPDFLoader("data/sample.pdf")
pdf_docs = pdf_loader.load()

print("PDF Loaded Successfully.\n")

# --------------------------------------------------
# Step 2 : Load HTML
# --------------------------------------------------

print("Loading HTML...")

html_loader = UnstructuredHTMLLoader("data/sample.html")
html_docs = html_loader.load()

print("HTML Loaded Successfully.\n")

# --------------------------------------------------
# Step 3 : Merge Documents
# --------------------------------------------------

documents = pdf_docs + html_docs

print("Total Documents :", len(documents))
print()

# --------------------------------------------------
# Step 4 : Recursive Character Splitting
# --------------------------------------------------

splitter = RecursiveCharacterTextSplitter(

    chunk_size=500,

    chunk_overlap=100,

    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]

)

recursive_chunks = splitter.split_documents(documents)

print("Recursive Character Splitting Completed.")
print("Number of Recursive Chunks :", len(recursive_chunks))
print()

# --------------------------------------------------
# Display Recursive Chunks
# --------------------------------------------------

print("=" * 70)
print("FIRST THREE RECURSIVE CHUNKS")
print("=" * 70)

for i, chunk in enumerate(recursive_chunks[:3]):

    print(f"\nChunk {i+1}\n")

    print(chunk.page_content)

    print("-" * 70)

# --------------------------------------------------
# Step 5 : Load Embedding Model
# --------------------------------------------------

print("\nLoading Sentence Transformer...\n")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model Loaded Successfully.\n")

# --------------------------------------------------
# Step 6 : Extract Sentences
# --------------------------------------------------

sentences = []

for doc in documents:

    text = doc.page_content

    temp = text.split(".")

    for sentence in temp:

        sentence = sentence.strip()

        if len(sentence) > 0:

            sentences.append(sentence)

print("Total Sentences :", len(sentences))
print()

# --------------------------------------------------
# Step 7 : Generate Embeddings
# --------------------------------------------------

print("Generating Sentence Embeddings...\n")

embeddings = model.encode(sentences)

print("Embedding Dimension :", len(embeddings[0]))
print()

# --------------------------------------------------
# Step 8 : Semantic Chunking
# --------------------------------------------------

semantic_chunks = []

current_chunk = [sentences[0]]

threshold = 0.70

for i in range(1, len(sentences)):

    similarity = cosine_similarity(

        [embeddings[i - 1]],

        [embeddings[i]]

    )[0][0]

    if similarity >= threshold:

        current_chunk.append(sentences[i])

    else:

        semantic_chunks.append(

            ". ".join(current_chunk)

        )

        current_chunk = [sentences[i]]

semantic_chunks.append(

    ". ".join(current_chunk)

)

print("Semantic Chunking Completed.")
print("Number of Semantic Chunks :", len(semantic_chunks))
print()

# --------------------------------------------------
# Display Semantic Chunks
# --------------------------------------------------

print("=" * 70)
print("FIRST FIVE SEMANTIC CHUNKS")
print("=" * 70)

for i, chunk in enumerate(semantic_chunks[:5]):

    print(f"\nChunk {i+1}\n")

    print(chunk)

    print("-" * 70)

# --------------------------------------------------
# Step 9 : Comparison
# --------------------------------------------------

print("\n")
print("=" * 70)
print("COMPARISON")
print("=" * 70)

print("Original Documents      :", len(documents))
print("Recursive Chunks        :", len(recursive_chunks))
print("Semantic Chunks         :", len(semantic_chunks))

print()

if len(recursive_chunks) > len(semantic_chunks):

    print("Semantic chunking produced fewer but more meaningful chunks.")

elif len(recursive_chunks) < len(semantic_chunks):

    print("Semantic chunking produced more granular chunks.")

else:

    print("Both strategies produced the same number of chunks.")

# --------------------------------------------------
# Step 10 : Preview Chunk Sizes
# --------------------------------------------------

print("\n")
print("=" * 70)
print("CHUNK SIZE COMPARISON")
print("=" * 70)

print()

print("Recursive Chunk Lengths")

for i, chunk in enumerate(recursive_chunks[:5]):

    print(

        f"Chunk {i+1} :",

        len(chunk.page_content),

        "characters"

    )

print()

print("Semantic Chunk Lengths")

for i, chunk in enumerate(semantic_chunks[:5]):

    print(

        f"Chunk {i+1} :",

        len(chunk),

        "characters"

    )

print()

print("=" * 70)
print("Semantic Data Ingestion Completed Successfully.")
print("=" * 70)