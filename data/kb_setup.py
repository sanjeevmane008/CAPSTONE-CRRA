"""
CRRA Lab C1 - Policy Knowledge Base Setup

Reads the BizOps procurement policy articles in data/kb/, splits each one into
one chunk per '## ' section, and loads them into a ChromaDB collection called
'crra_policy'. Then runs a few realistic BizOps questions to prove retrieval works.

Run from the project root:
    python data/kb_setup.py
"""

from pathlib import Path

import chromadb

KB_DIR = Path(__file__).resolve().parent / "kb"
COLLECTION_NAME = "crra_policy"

TEST_QUERIES = [
    "who approves a 60 lakh contract",
    "the contract auto-renews next month and we missed the notice deadline",
    "two monitoring tools that both have low licence usage",
    "vendor is asking for a 20 percent price increase at renewal",
    "we no longer need this tool at all, how do we exit and get our data out",
]


def chunk_markdown(text: str, filename: str) -> list[dict]:
    """Split one policy file into chunks at '## ' headings.

    Each heading covers one self-contained rule, so each rule becomes its own
    chunk. A whole file as a single chunk would bury the rule you want among
    unrelated rules and every search would return a weak match.
    """
    chunks = []
    heading = None
    body: list[str] = []

    def flush() -> None:
        text_body = "\n".join(body).strip()
        if heading and text_body:
            chunks.append({"heading": heading, "text": text_body})

    for line in text.splitlines():
        if line.startswith("## "):
            flush()
            heading = line[3:].strip()
            body = []
        elif line.startswith("# "):
            continue  # the document title, not a rule
        else:
            body.append(line)
    flush()

    return [
        {
            "id": f"{filename}::{i}",
            # Put the heading in the embedded text too: it is often the best
            # one-line summary of the rule and helps the match.
            "document": f"{c['heading']}\n{c['text']}",
            "metadata": {"source": filename, "heading": c["heading"]},
        }
        for i, c in enumerate(chunks)
    ]


def main() -> None:
    md_files = sorted(KB_DIR.glob("*.md"))
    if not md_files:
        raise SystemExit(f"No .md files found in {KB_DIR}. Check the folder path.")

    client = chromadb.Client()  # in-memory; nothing is written to disk

    # Delete and recreate so a re-run never stacks duplicate chunks.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # first run: nothing to delete
    # Cosine distance runs 0 (identical) to 2 (opposite), so 1 - distance gives
    # a readable confidence. ChromaDB's default (squared L2) does not, and can
    # make 1 - distance negative.
    collection = client.create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    ids, documents, metadatas = [], [], []
    print("Loading policy articles")
    print("=" * 60)
    for md_file in md_files:
        chunks = chunk_markdown(md_file.read_text(encoding="utf-8"), md_file.name)
        print(f"  {md_file.name:<32} {len(chunks):>3} chunks")
        for c in chunks:
            ids.append(c["id"])
            documents.append(c["document"])
            metadatas.append(c["metadata"])

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print("-" * 60)
    print(f"  TOTAL: {len(ids)} chunks from {len(md_files)} files\n")

    print("Testing retrieval (best match per question)")
    print("=" * 60)
    for query in TEST_QUERIES:
        result = collection.query(query_texts=[query], n_results=1)
        meta = result["metadatas"][0][0]
        confidence = 1 - result["distances"][0][0]
        print(f"  Q: {query}")
        print(f"     source:     {meta['source']}")
        print(f"     section:    {meta['heading']}")
        print(f"     confidence: {confidence:.2f}\n")

    print("=" * 60)
    print("KB ready. Lab C3's Analysis Agent will query this collection.")


if __name__ == "__main__":
    main()
