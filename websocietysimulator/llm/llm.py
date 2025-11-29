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


class GeminiLLM(LLMBase):
    """
    Google Gemini LLM wrapper.
    """
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        super().__init__(model)
        try:
            from google import genai
        except Exception as e:
            raise ImportError(
                "google.generativeai package is required for GeminiLLM. "
                "Install with `pip install google-generativeai` and follow "
                "Google Generative AI authentication instructions."
            ) from e

        try:
            if hasattr(genai, 'configure'):
                genai.configure(api_key=api_key)
            else:
                import os
                os.environ.setdefault('GOOGLE_API_KEY', api_key)
        except Exception:
            pass

        self.genai = genai

    def __call__(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 500, stop_strs: Optional[List[str]] = None, n: int = 1) -> Union[str, List[str]]:
        sdk = self.genai
        used_model = model or self.model

        response = None
        try:
            if hasattr(sdk, 'chat') and hasattr(sdk.chat, 'completions') and hasattr(sdk.chat.completions, 'create'):
                response = sdk.chat.completions.create(
                    model=used_model,
                    messages=messages,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    stop=stop_strs,
                    candidate_count=n
                )
            elif hasattr(sdk, 'chat') and hasattr(sdk.chat, 'create'):
                response = sdk.chat.create(
                    model=used_model,
                    messages=messages,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    stop=stop_strs,
                    candidate_count=n
                )
            elif hasattr(sdk, 'generate'):
                input_text = "\n".join(m['content'] for m in messages if 'content' in m)
                response = sdk.generate(model=used_model, prompt=input_text, max_output_tokens=max_tokens, temperature=temperature)
            else:
                raise RuntimeError("Unsupported google.generativeai SDK shape; please upgrade the SDK or adjust code.")
        except Exception as e:
            logger.error(f"Gemini call failed: {e}")
            raise

        def _extract_text(resp):
            try:
                if hasattr(resp, 'choices'):
                    choices = resp.choices
                    if len(choices) >= 1:
                        msg = getattr(choices[0], 'message', None)
                        if isinstance(msg, dict):
                            return msg.get('content') or msg.get('text')
                        if hasattr(msg, 'content'):
                            return msg.content
                if hasattr(resp, 'candidates'):
                    c = resp.candidates
                    if len(c) >= 1:
                        cand = c[0]
                        if isinstance(cand, dict):
                            return cand.get('content') or cand.get('text')
                        return getattr(cand, 'content', None) or getattr(cand, 'text', None)
                if isinstance(resp, dict):
                    if 'candidates' in resp and resp['candidates']:
                        return resp['candidates'][0].get('content') or resp['candidates'][0].get('text')
                    if 'output' in resp and resp['output']:
                        out = resp['output'][0]
                        if isinstance(out, dict):
                            return out.get('content') or out.get('text')
                out = getattr(resp, 'output', None)
                if out:
                    first = out[0]
                    return getattr(first, 'content', None) or (first.get('content') if isinstance(first, dict) else None)
            except Exception:
                return None
            return None

        if n == 1:
            text = _extract_text(response)
            if text is None:
                return str(response)
            return text
        else:
            results = []
            try:
                if hasattr(response, 'candidates'):
                    for cand in response.candidates[:n]:
                        if isinstance(cand, dict):
                            results.append(cand.get('content') or cand.get('text') or str(cand))
                        else:
                            results.append(getattr(cand, 'content', None) or getattr(cand, 'text', None) or str(cand))
                elif hasattr(response, 'choices'):
                    for choice in response.choices[:n]:
                        msg = getattr(choice, 'message', None)
                        if msg:
                            results.append(msg.get('content') if isinstance(msg, dict) else getattr(msg, 'content', None) or str(msg))
                else:
                    results.append(_extract_text(response) or str(response))
            except Exception:
                results = [str(response)]
            return results

    def get_embedding_model(self):
        return None
