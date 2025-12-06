import os
import re
from langchain_chroma import Chroma
from langchain.docstore.document import Document
import shutil
import uuid

class MemoryBase:
    def __init__(self, memory_type: str, llm) -> None:
        """
        Initialize the memory base class
        
        Args:
            memory_type: Type of memory
            llm: LLM instance used to generate memory-related text
        """
        self.llm = llm
        self.embedding = self.llm.get_embedding_model()
        db_path = os.path.join('./db', memory_type, f'{str(uuid.uuid4())}')
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
        self.scenario_memory = Chroma(
            embedding_function=self.embedding,
            persist_directory=db_path
        )

    def __call__(self, current_situation: str = ''):
        if 'review:' in current_situation:
            self.addMemory(current_situation.replace('review:', ''))
        else:
            return self.retriveMemory(current_situation)

    def retriveMemory(self, query_scenario: str):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def addMemory(self, current_situation: str):
        raise NotImplementedError("This method should be implemented by subclasses.")

class MemoryDILU(MemoryBase):
    def __init__(self, llm):
        super().__init__(memory_type='dilu', llm=llm)

    def retriveMemory(self, query_scenario: str):
        # Extract task name from query scenario
        task_name = query_scenario
        
        # Return empty string if memory is empty
        if self.scenario_memory._collection.count() == 0:
            return ''
            
        # Find most similar memory
        similarity_results = self.scenario_memory.similarity_search_with_score(
            task_name, k=1)
            
        # Extract task trajectories from results
        task_trajectories = [
            result[0].metadata['task_trajectory'] for result in similarity_results
        ]
        
        # Join trajectories with newlines and return
        return '\n'.join(task_trajectories)

    def addMemory(self, current_situation: str):
        # Extract task description
        task_name = current_situation
        
        # Create document with metadata
        memory_doc = Document(
            page_content=task_name,
            metadata={
                "task_name": task_name,
                "task_trajectory": current_situation
            }
        )
        
        # Add to memory store
        self.scenario_memory.add_documents([memory_doc])

class MemoryGenerative(MemoryBase):
    def __init__(self, llm):
        super().__init__(memory_type='generative', llm=llm)

    def retriveMemory(self, query_scenario: str):
        # Extract task name from query
        task_name = query_scenario
        
        # Return empty if no memories exist
        if self.scenario_memory._collection.count() == 0:
            return ''
            
        # Get top 3 similar memories
        similarity_results = self.scenario_memory.similarity_search_with_score(
            task_name, k=3)
            
        fewshot_results = []
        importance_scores = []

        # Score each memory's relevance
        for result in similarity_results:
            trajectory = result[0].metadata['task_trajectory']
            fewshot_results.append(trajectory)
            
            # Generate prompt to evaluate importance
#             prompt = f'''You will be given user reviews where you previously wrote really well, like a real user. Then you will be given an ongoing user review to write. Do not summarize these two previous reviews, but rather evaluate how relevant and helpful the previous reviews are for the ongoing review, on a scale of 1-10.
# Successful Review:
# {trajectory}
# Ongoing Review:
# {query_scenario}
# Your output format should be:
# Score: '''
            prompt = f'''You will be given a successful case where you successfully complete the task. Then you will be given an ongoing task. Do not summarize these two cases, but rather evaluate how relevant and helpful the successful case is for the ongoing task, on a scale of 1-10.
Success Case:
{trajectory}
Ongoing task:
{query_scenario}
Your output format should be:
Score: '''

            # Get importance score
            response = self.llm(messages=[{"role": "user", "content": prompt}], temperature=0.1, stop_strs=['\n'])
            score = int(re.search(r'\d+', response).group()) if re.search(r'\d+', response) else 0
            importance_scores.append(score)

        # Return trajectory with highest importance score
        max_score_idx = importance_scores.index(max(importance_scores))
        return similarity_results[max_score_idx][0].metadata['task_trajectory']
    
    def addMemory(self, current_situation: str):
        # Extract task description
        task_name = current_situation
        
        # Create document with metadata
        memory_doc = Document(
            page_content=task_name,
            metadata={
                "task_name": task_name,
                "task_trajectory": current_situation
            }
        )
        
        # Add to memory store
        self.scenario_memory.add_documents([memory_doc])

class MemoryTP(MemoryBase):
    def __init__(self, llm):
        super().__init__(memory_type='tp', llm=llm)

    def retriveMemory(self, query_scenario: str):
        # Extract task name from scenario
        task_name = query_scenario
        
        # Return empty if no memories exist
        if self.scenario_memory._collection.count() == 0:
            return ''
            
        # Find most similar memory
        similarity_results = self.scenario_memory.similarity_search_with_score(
            task_name, k=1)
            
        # Generate plans based on similar experiences
        experience_plans = []
        task_description = query_scenario
        
        for result in similarity_results:
            prompt = f"""You will be given a successful case where you successfully complete the task. Then you will be given an ongoing task. Do not summarize these two cases, but rather use the successful case to think about the strategy and path you took to attempt to complete the task in the ongoing task. Devise a concise, new plan of action that accounts for your task with reference to specific actions that you should have taken. You will need this later to solve the task. Give your plan after "Plan".
Success Case:
{result[0].metadata['task_trajectory']}
Ongoing task:
{task_description}
Plan:
"""
            experience_plans.append(self.llm(messaage=prompt, temperature=0.1))
            
        return 'Plan from successful attempt in similar task:\n' + '\n'.join(experience_plans)

    def addMemory(self, current_situation: str):
        # Extract task name
        task_name = current_situation
        
        # Create document with metadata
        memory_doc = Document(
            page_content=task_name,
            metadata={
                "task_name": task_name,
                "task_trajectory": current_situation
            }
        )
        
        # Add to memory store
        self.scenario_memory.add_documents([memory_doc])

class MemoryVoyager(MemoryBase):
    def __init__(self, llm):
        super().__init__(memory_type='voyager', llm=llm)

    def retriveMemory(self, query_scenario: str):
        # Extract task name from query
        task_name = query_scenario
        
        # Return empty if no memories exist
        if self.scenario_memory._collection.count() == 0:
            return ''
            
        # Find most similar memories
        similarity_results = self.scenario_memory.similarity_search_with_score(task_name, k=3)
        
        # Extract trajectories from results
        memory_trajectories = [result[0].metadata['task_trajectory'] 
                             for result in similarity_results]
                             
        return '\n'.join(memory_trajectories)

    def addMemory(self, current_situation: str):
        # Prompt template for summarizing trajectory
        voyager_prompt = '''You are a superbly helpful assistant that writes a description of the task resolution trajectory.

        1) Try to summarize the trajectory in no more than 5 sentences.
        2) Your response should be a single line of text.

        For example:

Retrieved user profile showing preference for Italian cuisine and budget-friendly options. Found restaurant with good ratings. Analyzed reviews emphasizing food quality and ambiance. Decided on 4-star rating with focus on authentic pasta and cozy atmosphere.

        Trajectory:
        '''
        
        # Generate summarized trajectory
        prompt = voyager_prompt + current_situation
        trajectory_summary = self.llm(messages=[{"role": "user", "content": prompt}], temperature=0.1)
        
        # Create document with metadata
        doc = Document(
            page_content=trajectory_summary,
            metadata={
                "task_description": trajectory_summary,
                "task_trajectory": current_situation
            }
        )
        
        # Add to memory store
        self.scenario_memory.add_documents([doc])


class MemoryHybrid(MemoryBase):
    def __init__(self, llm):
        super().__init__(memory_type='generative', llm=llm)

    def retriveMemory(self, query_scenario: str):
        # Extract task name from query
        task_name = query_scenario
        
        # Return empty if no memories exist
        if self.scenario_memory._collection.count() == 0:
            return ''
            
        # Get top 3 similar memories
        similarity_results = self.scenario_memory.similarity_search_with_score(
            task_name, k=3)
            
        fewshot_results = []
        importance_scores = []

        # Score each memory's relevance
        for result in similarity_results:
            trajectory = result[0].metadata['task_trajectory']
            fewshot_results.append(trajectory)
            
            # Generate prompt to evaluate importance
            prompt = f'''You will be given a successful case where you successfully complete the task. Then you will be given an ongoing task. Do not summarize these two cases, but rather evaluate how relevant and helpful the successful case is for the ongoing task, on a scale of 1-10.
Success Case:
{trajectory}
Ongoing task:
{query_scenario}
Your output format should be:
Score: '''
#             prompt = f'''You will be given user reviews where you previously wrote really well, like a real user. Then you will be given an ongoing user review to write. Do not summarize these two previous reviews, but rather evaluate how relevant and helpful the previous reviews are for the ongoing review, on a scale of 1-10.
# Successful Review:
# {trajectory}
# Ongoing Review:
# {query_scenario}
# Your output format should be:
# Score: '''

            # Get importance score
            response = self.llm(messages=[{"role": "user", "content": prompt}], temperature=0.1, stop_strs=['\n'])
            score = int(re.search(r'\d+', response).group()) if re.search(r'\d+', response) else 0
            importance_scores.append(score)

        # Return trajectory with highest importance score
        max_score_idx = importance_scores.index(max(importance_scores))
        return similarity_results[max_score_idx][0].metadata['task_trajectory']
    
    def addMemory(self, current_situation: str):
        # Prompt template for summarizing trajectory
        voyager_prompt = '''You are a superbly helpful assistant that writes a description of the task resolution trajectory.

        1) Try to summarize the trajectory in no more than 5 sentences.
        2) Your response should be a single line of text.

        For example:

Please fill in this part yourself

        Trajectory:
        '''
        
        # Generate summarized trajectory
        prompt = voyager_prompt + current_situation
        trajectory_summary = self.llm(messages=[{"role": "user", "content": prompt}], temperature=0.1)
        
        # Create document with metadata
        doc = Document(
            page_content=trajectory_summary,
            metadata={
                "task_description": trajectory_summary,
                "task_trajectory": current_situation
            }
        )
        
        # Add to memory store
        self.scenario_memory.add_documents([doc])

class MemoryIngrid(MemoryBase):
    """Hybrid memory: store short schema-like summaries (Voyager path) and
    re-rank with a fast generative pass. Retrieval returns a final
    contracted JSON produced by an LLM using an exemplar + compact digest.
    """
    def __init__(self, llm):
        super().__init__(memory_type='hybrid', llm=llm)

    def _parse_score_and_rationale(self, resp: str):
        """Parse a LLM response that should contain a line like "Score: <int>" and a short rationale.
        Returns (score:int, rationale:str). Uses conservative defaults on failure.
        """
        default_score = 5
        if not resp:
            return default_score, ''
        lines = [ln.strip() for ln in resp.splitlines() if ln.strip()]
        score = default_score
        rationale = ''
        for i, ln in enumerate(lines):
            low = ln.lower()
            if low.startswith('score:'):
                parts = ln.split(':', 1)
                if len(parts) > 1:
                    try:
                        score = int(parts[1].strip())
                    except Exception:
                        score = default_score
                # rationale may be on same line after score or next non-empty line
                remainder = parts[1].strip() if len(parts) > 1 else ''
                if remainder and not remainder.isdigit():
                    rationale = remainder
                else:
                    # pick next line if exists
                    if i+1 < len(lines):
                        rationale = lines[i+1]
                break

        score = max(0, min(5, score))
        return score, rationale

    def addMemory(self, current_situation: str):
        prompt = (
            "Write a 1-2 line compact schema summary of this trajectory. Include aspects (e.g. {food:+,service:-,price:+}), tone, rating_bias (numeric), and dealbreakers.\n\n"
            "Trajectory:\n"
        ) + current_situation

        try:
            summary = self.llm(messages=[{"role": "user", "content": prompt}], temperature=0.1)
        except Exception:
            summary = (current_situation.replace('\n', ' ')[:200]).strip()

        doc = Document(
            page_content=summary,
            metadata={
                'schema_summary': summary,
                'task_trajectory': current_situation
            }
        )
        self.scenario_memory.add_documents([doc])

    def retriveMemory(self, query_scenario: str):
        task_name = query_scenario
        if self.scenario_memory._collection.count() == 0:
            return ''

        results = self.scenario_memory.similarity_search_with_score(task_name, k=6)
        candidates = []
        for res in results:
            doc = res[0]
            rerank_prompt = (
                "Given task (user U, item I, known prefs S), rate usefulness 0-10.\n"
                "Candidate summary:\n" + doc.page_content + "\n\n"
                "Ongoing task:\n" + task_name + "\n\n"
                "Return lines starting with 'Score: <int>' and a one-line rationale."
            )
            try:
                resp = self.llm(messages=[{"role": "user", "content": rerank_prompt}], temperature=0.1)
            except Exception:
                resp = ''

            score, rationale = self._parse_score_and_rationale(resp)
            candidates.append({'doc': doc, 'score': score, 'rationale': rationale})

        if not candidates:
            return ''

        # sort and pick exemplar and a small set for digest
        candidates.sort(key=lambda c: c['score'], reverse=True)
        exemplar_doc = candidates[0]['doc']
        exemplar = exemplar_doc.metadata.get('task_trajectory', exemplar_doc.page_content)

        # pick up to two additional trajectories for context (keep it small)
        extra = [c['doc'].metadata.get('task_trajectory', c['doc'].page_content) for c in candidates[1:4]]

        # build digest prompt (compact)
        digest_prompt = (
            "Based on the exemplar and the additional short examples, produce a compact JSON with keys: aspects, tone, rating_bias, cautions. Keep it short.\n\n"
            "Exemplar:\n" + exemplar + "\n\n"
            "Extras:\n" + "\n\n".join(extra) + "\n\n"
            "JSON:" 
        )
        try:
            digest_resp = self.llm(messages=[{"role": "user", "content": digest_prompt}], temperature=0.1)
        except Exception:
            digest_resp = ''

        # # We keep digest generation internally for future use but do not parse
        # # or return structured JSON here to match other memory modules.
        # _ = digest_resp  # digest_resp intentionally unused for now
        return exemplar

