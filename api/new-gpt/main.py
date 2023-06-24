import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import traceback

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

@app.route('/api/new-gpt', methods=['POST'])
def my_endpoint():
    try:
        # Get the question from the request body
        question = request.json.get('question')

        if not question:
            return jsonify({'error': 'Question is missing in the request.'}), 400

        # Get the OpenAI API key from environment variable
        openai_api_key = os.getenv('OPENAI_API_KEY')

        if not openai_api_key:
            return jsonify({'error': 'OpenAI API key not found.'}), 500

        # Determine the file path
        file_path = 'EU-AI-ACT-2.txt'

        # Load the text documents
        loader = TextLoader(file_path, encoding='utf8')
        documents = loader.load()

        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        chat_history = []
        

        # Create the embeddings and Chroma index
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(texts, embeddings)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Set up the prompt template
        context = "The EU-AI-ACT is a regulatory framework governing AI development, deployment, and use in the European Union. It addresses ethics, transparency, accountability, and data protection. The goal is to ensure AI respects rights, promotes fairness, and manages risks. It includes articles on AI impact assessments, high-risk systems, human oversight, and public administration, fostering a responsible AI ecosystem."
        # prompt_template = """Please provide a concise, conversational, and contextually relevant response based on the EU-AI-ACT document. Prioritize information from the document, cite specific article numbers accurately, and use quotation marks when quoting text from the document. Use quotes as much as possible and keep three dots if the quoting exceeds 7 words. Keep the answer short while ensuring completeness. Maintain a professional and relatable tone throughout.If the input is irrelevant, encourage the user to ask a question explicitly related to the EU-AI-ACT. Conclude by suggesting the next appropriate step and provide a list of relevant articles. 
        prompt_template = """As an AI model, please read the EU AI Act document and provide one contextually relevant suggestion for each of the first four articles to ensure strict compliance. The aim is to make the AI systems comply with the EU AI Act document. Each suggestion should be specific to the particular use case described in the question and should mention the corresponding article number.

Article 1: Provide a suggestion for ensuring compliance with Article 1.
Article 2: Provide a suggestion for ensuring compliance with Article 2.
Article 3: Provide a suggestion for ensuring compliance with Article 3.
Article 4: Provide a suggestion for ensuring compliance with Article 4

        Context:
        {context}

        Question: {question}"""

        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        LLM = OpenAI(temperature=0.2,model_name="gpt-3.5-turbo")
        # Set up the RetrievalQA chain
        chain_type_kwargs = {"prompt": PROMPT}
        qa = RetrievalQA.from_chain_type(llm=LLM, chain_type="stuff", retriever=vectorstore.as_retriever(search_type="mmr", search_kwargs={"k":2}), chain_type_kwargs=chain_type_kwargs, memory=memory)

        # Run the query
        answer = qa.run(question)
        print("Answer:", answer)

        response = jsonify({'answer': answer})
        response.headers.add('Access-Control-Allow-Origin', '*')

        return response
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/check', methods=['GET'])
def check_endpoint():
    return jsonify({'message': 'API endpoint is working.'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
