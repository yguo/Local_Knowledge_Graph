from prompt_template import PROMPTS
import time
import llm
import re
import json

class ResponseGenerator:
    def __init__(self, db, llm, graph):
        self.db = db
        self.llm = llm
        self.graph = graph        

    def generate_response(self, user_prompt):
        messages = [
            {"role": "system", "content": PROMPTS["SYSTEM_MESSAGE"]},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": PROMPTS["INITIAL_RESPONSE"]}
        ]

        
        step_count = 1
        total_inference_time = 0
        final_answer = None

        while step_count < 20:
            step_json, inference_time = self._generate_step(messages)
            total_inference_time += inference_time
            if len(step_json['content']) > 700:
                messages.append({"role": "user", "content": PROMPTS["STEP_TOO_LONG"]})
                continue
            step_info = self._process_step(step_json, step_count)
            yield step_info

            if self._should_continue(step_json, step_count):
                messages.append({"role": "user", "content": PROMPTS["REQUEST_MORE_STEPS"].format(step_count=step_count-1)})
                continue
            if self._is_final_answer(step_json):
                final_answer = self._handle_final_answer(step_json, messages, user_prompt)
                if final_answer:
                    break
                else:
                    yield f"data: {json.dumps({'type': 'error', 'content': 'No final answer found after 5 steps of reasoning.   Restarting the reasoning process.'})}\n\n"
                    messages = self._reset_reasoning(messages)
                    step_count = 1
                    continue
            messages.append({"role": "assistant", "content": json.dumps(step_json)})
            step_count += 1

        if not final_answer:
            final_answer = self._generate_final_answer(messages)

        return self._finalize_response(final_answer, step_count, total_inference_time)
    
    def _get_short_title(self, content):
        messages = [
            {"role": "system", "content": PROMPTS["GET_SHORT_TITLE"]["SYSTEM"]},
            {"role": "user", "content": PROMPTS["GET_SHORT_TITLE"]["USER"].format(content=content[:100])}
        ]

        title_data = ""
        for chunk in self.llm.stream_api_call(messages, 50):
            title_data += chunk

        return title_data.strip()[:20]   

    def _generate_step(self, messages):
        start_time = time.time()
        step_data = ""
        for chunk in self.llm.stream_api_call(messages, max_tokens=300):    
            step_data += chunk
        end_time = time.time()
        inference_time = end_time - start_time
        return self._extract_json(step_data), inference_time

    def _extract_json(self, text):
        text = re.sub(r'```(?:json)?\s*', '', text)
        text = text.strip()
        json_objects = re.findall(r'\{[^{}]*\}', text)

        if json_objects:
            try:
                return json.loads(json_objects[-1])
            except json.JSONDecodeError:
                pass

        return {
            "title": "Parsing Error",
            "content": text,
            "next_action": "continue"
        }
    
    def _process_step(self, step_json, step_count):
        embedding = self.llm.get_embedding(step_json['content'])
        self.db.insert_data(step_json['content'], embedding, False, id=f"Step{step_count}")

        short_title = self._get_short_title(step_json['content']) if not step_json['title'] or len(step_json['title']) > 20 else step_json['title'][:20]

        node_id = f"Step{step_count}"
        self.graph.add_node(node_id, f"Step {step_count}: {short_title}")

        if step_count > 1:
            self._update_graph_edges(node_id, embedding)

        return self._format_step_data(step_json, step_count)
    
    def _should_continue(self, step_json, step_count):
        return step_json['next_action'] == 'final_answer' and step_count <= 5

    def _is_final_answer(self, step_json):
        return step_json['next_action'] == 'final_answer' or 'boxed' in step_json['content'].lower()

    def _handle_final_answer(self, step_json, messages, prompt):
        messages.append({"role": "user", "content": PROMPTS["FINAL_EVALUATION"].format(prompt=prompt)})
        evaluation_json, _ = self._generate_step(messages)
        if self._check_consistency(step_json['content'], evaluation_json['content']):
            return step_json['content']
        return None

    def _reset_reasoning(self, messages):
        return messages[:2]  # Keep only the system message and the original user prompt

    def _generate_final_answer(self, messages):
        messages.append({"role": "user", "content": PROMPTS["GENERATE_FINAL_ANSWER"]})
        final_json, _ = self._generate_step(messages)
        return final_json.get('content', '')
    
    def _update_graph_edges(self, node_id, current_embedding):
        top_similarities = self._calculate_top_similarities(current_embedding, top_k=2)
        self.graph.edge_dict = {k: v for k, v in self.graph.edge_dict.items() if v['to'] != node_id}

        for prev_node_id, similarity in top_similarities:
            if prev_node_id in [node['id'] for node in self.graph.graph_data['nodes']]:
                self.graph.add_edge(prev_node_id, node_id, similarity, 300 * (1 - similarity))

    def _calculate_top_similarities(self, current_embedding, top_k=2):
        results = self.db.collection.query(
            query_embeddings=[current_embedding.tolist()],
            n_results=top_k + 1  # +1 because the current embedding will be included
        )
        
        similarities = []
        for i, (distance, id) in enumerate(zip(results['distances'][0], results['ids'][0])):
            if id != f"Step{len(self.graph.graph_data['nodes'])}":  # Exclude the current step
                similarity = 1 - distance  # Convert distance to similarity
                similarities.append((id, similarity))
        
        return similarities[:top_k]
    
    def _format_step_data(self, step_json, step_count):
        serialized_graph_data = self.graph.serialize_graph_data()
        strongest_path, path_weights, avg_similarity = self.graph.calculate_strongest_path(step_count)

        path_data = {
            'strongest_path': strongest_path,
            'path_weights': path_weights,
            'avg_similarity': avg_similarity
        } if strongest_path is not None else None

        return f"data: {json.dumps({'type': 'step', 'step': step_count, 'title': step_json['title'], 'content': step_json['content'], 'graph': serialized_graph_data, 'path_data': path_data})}\n\n"
    
    def _finalize_response(self, final_answer, step_count, total_thinking_time):
        final_embedding = self.llm.get_embedding(final_answer)
        self.db.insert_data(final_answer, final_embedding, False, id=f"Step{step_count}")

        final_node_id = f"Step{step_count}"
        self.graph.add_node(final_node_id, f"Final Answer: {self.get_short_title(final_answer)}")

        self._update_graph_edges(final_node_id, final_embedding)

        serialized_graph_data = self.graph.serialize_graph_data()
        strongest_path, path_weights, avg_similarity = self.graph.calculate_strongest_path(step_count)

        path_data = {
            'strongest_path': strongest_path,
            'path_weights': path_weights,
            'avg_similarity': avg_similarity
        } if strongest_path is not None else None

        yield f"data: {json.dumps({'type': 'final', 'content': final_answer, 'graph': serialized_graph_data, 'path_data': path_data})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'total_time': total_thinking_time})}\n\n"
    
    def _check_consistency(self, final_answer, evaluation):
        return True

