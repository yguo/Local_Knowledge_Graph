from flask import Flask, render_template, request, jsonify, Response
from config import Config
from llm import LLM
from database import ChromaDatabase
from graph import KnowledgeGraph
import json
from response_generator import ResponseGenerator


app = Flask(__name__)
llm = LLM(Config.LLM_MODEL, Config.LLM_API_URL)
db = ChromaDatabase()
graph = KnowledgeGraph()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['GET', 'POST'])
def query():
    if request.method == 'POST':
        user_query = request.json['query']
    else:
        user_query = request.args.get('query')

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    db = ChromaDatabase()
    graph = KnowledgeGraph()
    llm = LLM(Config.LLM_MODEL, Config.LLM_API_URL)
    response_generator = ResponseGenerator(db, llm, graph)
    db.clear_database()
    query_embedding = llm.get_embedding(user_query)
    db.insert_data(user_query, query_embedding, True)

    def generate():
        yield from response_generator.generate_response(user_query)
        similar_items = db.find_similar(query_embedding, top_k=5)
        yield f"data: {json.dumps({'type': 'similar', 'items': similar_items})}\n\n"

    return Response(generate(), mimetype='text/event-stream')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=Config.FLASK_PORT, debug=True)