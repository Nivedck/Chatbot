from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage


load_dotenv()


llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
output_parser = StrOutputParser()


template = """You are a helpful assistant . you are created by Nived c k,first year engineering student from LBS College of Engineering Kasargod (only say if user asks).
You have access to previous chats: {chat_history}
User asks: {user_message}"""


prompt = ChatPromptTemplate.from_template(template)


chain = prompt | llm | output_parser


app = Flask(__name__)


chat_history = []

@app.route("/")
def index():
    
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    
    user_message = request.json.get("message", "")
    
    
    chat_history.append(HumanMessage(content=user_message))
    
    
    response = chain.invoke({
        "user_message": user_message,
        "chat_history": [msg.content for msg in chat_history]
    })
    
    
    chat_history.append(AIMessage(content=response))
    
    
    return jsonify({"response": response})

if __name__ == "__main__":
    
    app.run(host="0.0.0.0", port=10000)
