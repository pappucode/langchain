from langchain_community.document_loaders import AmazonTextractPDFLoader

loader = AmazonTextractPDFLoader('test_page.jpeg')

docs = loader.load()

print(len(docs))
#print(docs[0])
print(docs[0].page_content)
print(docs[0].metadata)
