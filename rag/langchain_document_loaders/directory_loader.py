from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
    )

docs = loader.lazy_load()

#print(len(docs))
# print(docs[1].page_content)
# print(docs[1].metadata)

for document in docs:
    print(document.metadata)

