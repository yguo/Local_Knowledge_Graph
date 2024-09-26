import requests
import json
import numpy as np

class LLM:
    def __init__(self, model, api_url):
        self.model = model
        self.api_url = api_url

    def get_embedding(self, text):
        headers = {'Content-Type': 'application/json'}
        data = json.dumps({"model": self.model, "input": text})
        response = requests.post(f'{self.api_url}/api/embed', headers=headers, data=data)
        
        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
        
        response_data = response.json()
        
        if 'embedding' in response_data:
            return np.array(response_data['embedding'], dtype=np.float32)
        elif 'embeddings' in response_data and response_data['embeddings']:
            return np.array(response_data['embeddings'][0], dtype=np.float32)
        else:
            raise KeyError(f"No embedding found in API response. Response: {response_data}")

    def stream_api_call(self, messages, max_tokens):
        prompt = json.dumps(messages)
        data = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "stream": True
        }
        try:
            response = requests.post(f'{self.api_url}/api/generate', 
                                     headers={'Content-Type': 'application/json'}, 
                                     data=json.dumps(data),
                                     stream=True)
            response.raise_for_status()
            full_response = ""
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode('utf-8'))
                    if 'response' in chunk:
                        full_response += chunk['response'].replace("'",'')
                        yield chunk['response'].replace("'",'')
            if full_response:
                return json.loads(full_response.replace("'",''))
            else:
                raise ValueError("Empty response from API")
        except Exception as e:
            error_message = f"Failed to generate response. Error: {str(e)}"
            return {"title": "Error", "content": error_message, "next_action": "final_answer"}