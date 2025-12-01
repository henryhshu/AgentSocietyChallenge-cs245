from typing import Dict, List, Optional, Union
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from .infinigence_embeddings import InfinigenceEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging
logger = logging.getLogger("websocietysimulator")

class LLMBase:
    def __init__(self, model: str = "qwen2.5-72b-instruct"):
        """
        Initialize LLM base class
        
        Args:
            model: Model name, defaults to deepseek-chat
        """
        self.model = model
        
    def __call__(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 500, stop_strs: Optional[List[str]] = None, n: int = 1) -> Union[str, List[str]]:
        """
        Call LLM to get response
        
        Args:
            messages: List of input messages, each message is a dict containing role and content
            model: Optional model override
            max_tokens: Maximum tokens in response, defaults to 500
            stop_strs: Optional list of stop strings
            n: Number of responses to generate, defaults to 1
            
        Returns:
            Union[str, List[str]]: Response text from LLM, either a single string or list of strings
        """
        raise NotImplementedError("Subclasses need to implement this method")
    
    def get_embedding_model(self):
        """
        Get the embedding model for text embeddings
        
        Returns:
            OpenAIEmbeddings: An instance of OpenAI's text embedding model
        """
        raise NotImplementedError("Subclasses need to implement this method")

class InfinigenceLLM(LLMBase):
    def __init__(self, api_key: str, model: str = "qwen2.5-72b-instruct"):
        """
        Initialize Deepseek LLM
        
        Args:
            api_key: Deepseek API key
            model: Model name, defaults to qwen2.5-72b-instruct
        """
        super().__init__(model)
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://cloud.infini-ai.com/maas/v1"
        )
        self.embedding_model = InfinigenceEmbeddings(api_key=api_key)
        
    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=10, max=300),  # 等待时间从10秒开始，指数增长，最长300秒
        stop=stop_after_attempt(10)  # 最多重试10次
    )
    def __call__(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 500, stop_strs: Optional[List[str]] = None, n: int = 1) -> Union[str, List[str]]:
        """
        Call Infinigence AI API to get response with rate limit handling
        
        Args:
            messages: List of input messages, each message is a dict containing role and content
            model: Optional model override
            max_tokens: Maximum tokens in response, defaults to 500
            stop_strs: Optional list of stop strings
            n: Number of responses to generate, defaults to 1
            
        Returns:
            Union[str, List[str]]: Response text from LLM, either a single string or list of strings
        """
        try:
            response = self.client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_strs,
                n=n,
            )
            
            if n == 1:
                return response.choices[0].message.content
            else:
                return [choice.message.content for choice in response.choices]
        except Exception as e:
            if "429" in str(e):
                logger.warning("Rate limit exceeded")
            else:
                logger.error(f"Other LLM Error: {e}")
            raise e
    
    def get_embedding_model(self):
        return self.embedding_model

class OpenAILLM(LLMBase):
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Initialize OpenAI LLM
        
        Args:
            api_key: OpenAI API key
            model: Model name, defaults to gpt-3.5-turbo
        """
        super().__init__(model)
        self.client = OpenAI(api_key=api_key)
        self.embedding_model = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")
        
    def __call__(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 500, stop_strs: Optional[List[str]] = None, n: int = 1) -> Union[str, List[str]]:
        """
        Call OpenAI API to get response
        
        Args:
            messages: List of input messages, each message is a dict containing role and content
            model: Optional model override
            max_tokens: Maximum tokens in response, defaults to 500
            stop_strs: Optional list of stop strings
            n: Number of responses to generate, defaults to 1
            
        Returns:
            Union[str, List[str]]: Response text from LLM, either a single string or list of strings
        """
        response = self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_strs,
            n=n
        )
        
        if n == 1:
            return response.choices[0].message.content
        else:
            return [choice.message.content for choice in response.choices]
    
    def get_embedding_model(self):
        return self.embedding_model 


from typing import List, Dict, Optional, Union

import google.genai as genai
from google.genai import types as genai_types
import os


class GeminiLLM(LLMBase):
    """
    Google Gemini LLM wrapper using the google-genai SDK.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        super().__init__(model)

        # This is the *only* place the client is created.
        # API key is passed explicitly here.

        self.client = genai.Client(api_key=api_key)


    def __call__(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        stop_strs: Optional[List[str]] = None,
        n: int = 1,
    ) -> Union[str, List[str]]:

        used_model = model or self.model

        # Convert OpenAI-style messages into Gemini Content objects
        # messages: [{"role": "user"|"assistant"|"system", "content": "..."}]
        contents: List[genai_types.Content] = []
        for m in messages:
            text = m.get("content", "")
            if not text:
                continue
            role = m.get("role", "user")
            contents.append(
                genai_types.Content(
                    role=role,
                    parts=[genai_types.Part.from_text(text=text)],
                )
            )

        # Build generation config
        gen_config = genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            stop_sequences=stop_strs or [],
            candidate_count=n,
        )

        try:
            resp = self.client.models.generate_content(
                model=used_model,
                contents=contents,
                config=gen_config,
            )
        except Exception as e:
            logger.error(f"Gemini call failed: {e}")
            raise

        # Helper to extract one string from the response
        def _extract_single_text(r) -> str:
            # The SDK provides resp.text as a convenience for the first candidate
            txt = getattr(r, "text", None)
            if txt:
                return txt

            # Fallback: manually join candidate parts if needed
            candidates = getattr(r, "candidates", None) or []
            if candidates:
                cand = candidates[0]
                content = getattr(cand, "content", None)
                if content and getattr(content, "parts", None):
                    parts = content.parts
                    texts = [
                        getattr(p, "text", "")
                        for p in parts
                        if hasattr(p, "text") and getattr(p, "text", None)
                    ]
                    if texts:
                        return "\n".join(texts)

            # Last resort: string cast
            return str(r)

        # If n == 1, just return one string
        if n == 1:
            return _extract_single_text(resp)

        # For n > 1, extract each candidate
        results: List[str] = []
        candidates = getattr(resp, "candidates", None) or []
        for cand in candidates[:n]:
            # Try cand.content.parts[..].text
            content = getattr(cand, "content", None)
            if content and getattr(content, "parts", None):
                texts = [
                    getattr(p, "text", "")
                    for p in content.parts
                    if hasattr(p, "text") and getattr(p, "text", None)
                ]
                if texts:
                    results.append("\n".join(texts))
                    continue
            # Fallback: cand.text or str(cand)
            txt = getattr(cand, "text", None)
            results.append(txt if txt else str(cand))

        if not results:
            # fallback: at least return something
            results = [_extract_single_text(resp)]
        return results

    def get_embedding_model(self):
        """
        Return a callable embedding function for Chroma/other vector stores,
        backed by Gemini embeddings via google-genai.
        """

        client = self.client  # reuse the same client

        class _GenAIEmbeddings:
            """
            Adapter exposing embed_documents/embed_query backed by Gemini embeddings.

            Uses client.models.embed_content(model=..., contents=[...]).
            """

            def __init__(self, client, model_name: str = "text-embedding-004"):
                self._client = client
                self._model_name = model_name

            def _embed_batch(self, texts: List[str]) -> List[List[float]]:
                if not texts:
                    return []

                resp = self._client.models.embed_content(
                    model=self._model_name,
                    contents=texts,
                )

                # resp.embeddings: list of embedding objects, each with .values
                embeddings = getattr(resp, "embeddings", None)
                if embeddings is None and isinstance(resp, dict):
                    embeddings = resp.get("embeddings")

                if not embeddings:
                    return []

                vectors: List[List[float]] = []
                for emb in embeddings:
                    if emb is None:
                        continue
                    vals = getattr(emb, "values", None)
                    if vals is None and isinstance(emb, dict):
                        vals = emb.get("values")
                    if vals is None:
                        continue
                    vectors.append(list(vals))

                return vectors

            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                return self._embed_batch(texts)

            def embed_query(self, text: str) -> List[float]:
                vecs = self._embed_batch([text])
                return vecs[0] if vecs else []

            async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
                import asyncio
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, self.embed_documents, texts)

            async def aembed_query(self, text: str) -> List[float]:
                vecs = await self.aembed_documents([text])
                return vecs[0] if vecs else []

        # You can change the default embedding model name if you want
        return _GenAIEmbeddings(client, model_name="text-embedding-004")