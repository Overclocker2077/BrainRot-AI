from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
import os
import API_KEY
import logging
# from TTS.api import TTS
# tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(filename='Chatbot.log', encoding='utf-8', level=logging.DEBUG)
logger.info(msg="INITIALIZING PROGRAM!!!!")

# Set OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = API_KEY.API_KEY()

class ChatBot():
    def __init__(self, persist_dir: str):
        self.SYSTEM_PROMPT = ChatPromptTemplate.from_template(""" Prompt Not Included
                                                          <context>{context}</context>
                                                          User Question: {questionx}
                                                          """)
                                                        
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.embedding_model = OpenAIEmbeddings()

        self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        self.chain = self.SYSTEM_PROMPT | self.llm | StrOutputParser()

        self.persist_dir = persist_dir

    # Only works with PDF files TODO: Make txt files compatible
    def create_vector_db(self, directory) -> None:
        docs = self.text_splitter.split_documents(PyPDFLoader(directory).load())
        # Debug: Print number of documents loaded
        print(f"Number of documents loaded: {len(docs)}")
        logger.info(f"Number of documents loaded: {len(docs)}")
        # Use embedding model to embed the documents into a vector database
        self.vector_db = FAISS.from_documents(documents=docs, embedding=self.embedding_model)
        self.vector_db.save_local(self.persist_dir)

    # This loads a vector db and also saves it to the disk
    def load_vector_db(self) -> None:
        try:
            self.vector_db = FAISS.load_local(self.persist_dir, self.embedding_model, allow_dangerous_deserialization=True)
            logger.info("Loading vector database was successful!!")
        except Exception as e:
            logger.error(msg=f"Failed to load vector database: {e}")
            raise

    def get_response(self, message: str):
        docs = self.vector_db.similarity_search(query=message, k=5)
        logger.warning(msg=f"#### ----> Document list len: {len(docs)}")
        # print(f"Number of similar documents found: {len(docs)}")
        context = "\n".join([doc.page_content for doc in docs])
        return self.chain.invoke({"context": context, "questionx": message})

# File path to the PDF
file_path = "docs/BrainRot AI.pdf"
persist_dir = "ProjectBrainRotAI"
chatbot = ChatBot(persist_dir)

# Create the vector database if it does not exist
if not os.path.exists(persist_dir):
    chatbot.create_vector_db(file_path)

chatbot.load_vector_db()
print('\n')
while True:
    os.system("cls")
    user_input = input("Enter a prompt: ")
    r = chatbot.get_response(user_input)
    print(r)
    # logger.info(msg=r)
    input()

# generate speech by cloning a voice using default settings
# tts.tts_to_file(text=r,
#                 file_path="output.wav",
#                 speaker_wav="speaker.wav",
#                 language="en")
