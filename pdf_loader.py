from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader

def load_pdf(pdf_path):
    loader=DirectoryLoader(pdf_path,glob='*.pdf',loader_cls=PyPDFLoader)
    document=loader.load()
    return document