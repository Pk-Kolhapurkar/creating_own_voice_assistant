import os
import json
from abc import ABC, abstractmethod
from groq import Groq as Groqq, AsyncGroq as AsyncGroqq

# Check if the GROQ_API_KEY environment variable is set
api_key = os.getenv("GROQ_API_KEY", "gsk_VTwOqj6teSThrYLUCchlWGdyb3FYc6JB5LinaLL5JvriCPUVvGZk")  # Default API key if environment variable is not set

class LLM(ABC):
    def __init__(self, model, temperature=0.7, max_tokens=100, top_p=0.9, stream=True, model_params=None):
        self.model = model
        self.intent_model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.stream = stream
        self.client = Groqq(api_key=api_key)  # Use the provided API key for Groq client

        self.llm_params = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": stream,
        }
        
        self.llm_params.update(model_params or {})

    def completion(self, messages, **kwargs):
        raise NotImplementedError("Method not implemented")
        pass

    async def acompletion(self, messages, **kwargs):
        raise NotImplementedError("Method not implemented")
        pass
    
    def update_params(self, **kwargs):
        self.llm_params.update(kwargs)
        
    def intentify(self, text):
        completion = self.client.chat.completions.create(
            model=self.intent_model,
            messages=[{
                    "role": "system",
                    "content": "You always respond in JSON. "
                },
                {
                    "role": "user",
                    "content": "You always respond in JSON. Your task is to do a binary classification for the given query. Classify the given query into one of the following three classes:\n\n1/ \"QUESTION\": When the user's message expects an answer.\n2/ \"PARTIAL\": When the message is grammatically incomplete.\n3/ \"DEFAULT\": When it's a casual statement and no response is needed.\n\nOutput format:\n{\"class\":\"[Class Label]\"}\n\nQuery:\n" + text
                }
            ],
            temperature=0.01,
            max_tokens=50,
            top_p=1,
            stream=False,
            response_format={"type": "json_object"},
            stop=None,
        )

        response = completion.choices[0].message.content
        return json.loads(response).get("class", "DEFAULT")

class Groq(LLM):
    def __init__(self, model="llama3-8b-8192", **kwargs):
        super().__init__(model, **kwargs)
        self.client = Groqq(api_key=api_key)  # Ensure the API key is passed here
        self.intent_model = "llama3-70b-8192"
    
    def completion(self, messages, **kwargs):
        # Update self.llm_params with kwargs
        self.llm_params.update(kwargs)
        
        return self.client.chat.completions.create(
            messages=messages,
            **self.llm_params,
        )
        
    def acompletion(self, messages, **kwargs):
        # Raise error if this method is not implemented
        raise NotImplementedError("Method not implemented")

class AsyncGroq(LLM):
    def __init__(self, model="llama3-8b-8192", **kwargs):
        super().__init__(model, **kwargs)
        self.client = AsyncGroqq(api_key=api_key)  # Ensure the API key is passed here
        self.intent_model = "llama3-70b-8192"
    
    async def acompletion(self, messages, **kwargs):
        # Update self.llm_params with kwargs
        self.llm_params.update(kwargs)
    
        return await self.client.chat.completions.create(
            messages=messages,
            **self.llm_params,
        )

class Ollama(Groq):
    def __init__(self, model="llama3", **kwargs):
        super().__init__(model, **kwargs)
        self.client.base_url = "http://localhost:11434/v1"
        self.client.api_key = "ollama"

class AsyncOllama(AsyncGroq):
    def __init__(self, model="llama3", **kwargs):
        super().__init__(model, **kwargs)
        self.client.base_url = "http://localhost:11434/v1"
        self.client.api_key = "ollama"

if __name__ == "__main__":
    client = Groq()  # Initialize the Groq client with your API key
    stream = client.completion(
                messages=[{"role": "user", "content": "Why is the sky blue?"}],
                stream=True
            )
    
    # Iterate through the stream and print responses
    for chunk in stream:
        response_text = chunk.choices[0].delta.content
        print(response_text)
