from flask import Flask, render_template, request, jsonify, Response
from .config import Config
from .llm import LLM
from .database import ChromaDatabase
from .graph import KnowledgeGraph
import json
from .response_generator import ResponseGenerator
import os


class App:
    def __init__(self):
        template_dir = os.path.abspath('src/frontend/templates')
        self.app = Flask(__name__, template_folder=template_dir)
        self.llm = LLM(Config.LLM_MODEL, Config.LLM_API_URL)
        self.db = ChromaDatabase()
        self.graph = KnowledgeGraph()
        self.setup_routes()
    
    def setup_routes(self):
        self.app.route('/')(self.index)
        self.app.route('/query', methods=['GET', 'POST'])(self.query)
 
    
    def index(self):
        return render_template('index.html')

    
    def query(self):
        if request.method == 'POST':
            user_query = request.json['query']
        else:
            user_query = request.args.get('query')

        if not user_query:
            return jsonify({"error": "No query provided"}), 400

        self.db.clear_database()

        self.response_generator = ResponseGenerator(self.db, self.llm, self.graph)
        self.db.clear_database()
        query_embedding = self.llm.get_embedding(user_query)
        self.db.insert_data(user_query, query_embedding, True)

        def generate():
            yield from self.response_generator.generate_response(user_query)
            similar_items = self.db.find_similar(query_embedding, top_k=5)
            yield f"data: {json.dumps({'type': 'similar', 'items': similar_items})}\n\n"

        return Response(generate(), mimetype='text/event-stream')

    def run(self, host='0.0.0.0', port = Config.FLASK_PORT, debug=True):
        self.app.run(host=host, port=port, debug=debug)