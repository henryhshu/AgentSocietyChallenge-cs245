from collections import Counter
import re
import json

class ReasoningBase:
    def __init__(self, profile_type_prompt, memory, llm, use_planning_input: bool = True):
        """
        Initialize the reasoning base class
        
        Args:
            profile_type_prompt: Profile type prompt
            memory: Memory module
            llm: LLM instance used to generate reasoning
            use_planning_input: whether the reasoner should accept planner output
                                as structured input (default: True)
        """
        self.profile_type_prompt = profile_type_prompt
        self.memory = memory
        self.llm = llm
        # flag that controls whether planner output (JSON/list) should be
        # interpreted/consumed by reasoning classes that support it.
        self.use_planning_input = bool(use_planning_input)
    
    def process_task_description(self, task_description):
        examples = ''
        return examples, task_description

class ReasoningIO(ReasoningBase):
    def __call__(self, task_description: str, feedback :str= ''):
        examples, task_description = self.process_task_description(task_description)
        prompt = '''Your instructions must follow the examples.
Here are some examples.
{examples}
Here is the task:
{task_description}'''
        prompt = prompt.format(task_description=task_description, examples=examples)
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
        )
        
        return reasoning_result
    
class ReasoningCOT(ReasoningBase):
    def __call__(self, task_description: str, feedback :str= ''):
        examples, task_description = self.process_task_description(task_description)
        prompt = '''Solve the task step by step. Your instructions must follow the examples.
Here are some examples.
{examples}
Here is the task:
{task_description}'''
        prompt = prompt.format(task_description=task_description, examples=examples)
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
        )
        return reasoning_result

class ReasoningCOTSC(ReasoningBase):
    def __call__(self, task_description: str, feedback :str= ''):
        examples, task_description = self.process_task_description(task_description)
        prompt = '''Solve the task step by step. Your instructions must follow the examples.
Here are some examples.
{examples}
Here is the task:
{task_description}'''
        prompt = prompt.format(task_description=task_description, examples=examples)
        messages = [{"role": "user", "content": prompt}]
        reasoning_results = self.llm(
            messages=messages,
            temperature=0.1,
            n=5
        )
        string_counts = Counter(reasoning_results)
        reasoning_result = string_counts.most_common(1)[0][0]
        return reasoning_result
    
class ReasoningTOT(ReasoningBase):
    def __call__(self, task_description: str, feedback :str= ''):
        examples, task_description = self.process_task_description(task_description)
        prompt = '''Solve the task step by step. Your instructions must follow the examples.
Here are some examples.
{examples}
Here is the task:
{task_description}'''
        prompt = prompt.format(task_description=task_description, examples=examples)
        messages = [{"role": "user", "content": prompt}]
        reasoning_results = self.llm(
            messages=messages,
            temperature=0.1,
            n=3
        )
        reasoning_result = self.get_votes(task_description, reasoning_results, examples)
        return reasoning_result
    def get_votes(self, task_description, reasoning_results, examples):
        if 'think'  in reasoning_results[0].lower():
            return reasoning_results[0]
        prompt = '''Given the reasoning process for two completed tasks and one ongoing task, and several answers for the next step, decide which answer best follows the reasoning process for example command format. Output "The best answer is {{s}}", where s is the integer id chosen.
Here are some examples.
{examples}
Here is the task:
{task_description}

'''     
        prompt = prompt.format(task_description=task_description, examples=examples)
        messages = [{"role": "user", "content": prompt}]
        for i, y in enumerate(reasoning_results, 1):
            prompt += f'Answer {i}:\n{y}\n'
        vote_outputs = self.llm(
            messages=messages,
            temperature=0.7,
            n=5
        )
        vote_results = [0] * len(reasoning_results)
        for vote_output in vote_outputs:
            pattern = r".*best answer is .*(\d+).*"
            match = re.match(pattern, vote_output, re.DOTALL)
            if match:
                vote = int(match.groups()[0]) - 1
                if vote in range(len(reasoning_results)):
                    vote_results[vote] += 1
            else:
                print(f'vote no match: {[vote_output]}')
        ids = list(range(len(reasoning_results)))
        select_id = sorted(ids, key=lambda x: vote_results[x], reverse=True)[0]
        return reasoning_results[select_id]

class ReasoningDILU(ReasoningBase):
    def __call__(self, task_description: str, feedback :str= ''):
        examples, task_description = self.process_task_description(task_description)
        messages = [
            {
                "role": "system",
                "content": '''You must act as a real human user on Yelp. You will be given a detailed description of the scenario of current frame along with your history of previous decisions. 
'''
            },
            {
                "role": "user",
                "content": f'''Above messages are some examples of how you make a step successfully in the past. Those scenarios are similar to the current scenario. You should refer to those examples to make a step for the current scenario. Your instructions must follow the examples.
Here are two examples.
{examples}
Here is the task:
{task_description}'''
            }
        ]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
        )
        return reasoning_result

class ReasoningSelfRefine(ReasoningBase):
    def __call__(self, task_description: str, feedback :str= ''):
        examples, task_description = self.process_task_description(task_description)
        prompt = '''Solve the task step by step. Your instructions must follow the examples.
Here are some examples.
{examples}
Here is the task:
{task_description}'''
        prompt = prompt.format(task_description=task_description, examples=examples)
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
        )
        reasoning_result = self.refine(reasoning_result)
        return reasoning_result
    def refine(self, reasoning_result):
        prompt = f'''Reflect on the reasoning process and identify any potential errors or areas for improvement. Provide a revised version of the reasoning if necessary.
Here is the original reasoning:
{reasoning_result}
'''     
        messages = [{"role": "user", "content": prompt}]
        feedback_result = self.llm(
            messages=messages,
            temperature=0.0
        )
        return feedback_result
        
class ReasoningStepBack(ReasoningBase):
    def __call__(self, task_description: str, feedback :str= ''):
        examples, task_description = self.process_task_description(task_description)
        self.principle = self.stepback(task_description)
            
        prompt = f'''Solve the task step by step. Your instructions must follow the examples.
Here are some examples.
{examples}{self.principle}
Here is the task:
{task_description}'''
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
        )
        return reasoning_result
    def stepback(self, task_description):
        stepback_prompt = f'''What common sense, instruction structure is involved in solving this task?
{task_description}'''
        messages = [{"role": "user", "content": stepback_prompt}]
        principle = self.llm(
            messages=messages,
            temperature=0.1,
        )
        return principle


class ReasoningTemplateDrafting(ReasoningBase):
    """Reasoner that consumes a planner output produced by
    `PlanningTemplateDrafting` (a JSON array of step objects) and produces
    concrete drafts for each subtask using memories and the LLM.

    Behavior:
    - Accepts either a Python list of dicts or a JSON string representing
      the planner output.
    - For each plan item, retrieves relevant memories using `self.memory` and
      asks `self.llm` to fill the provided `draft_template` according to the
      `reasoning_instruction` and memory context.
    - Returns a list of dicts with keys: `description`, `tool_use_instruction`,
      `draft`, and `format`.
    """

    def __call__(self, task_description: str, feedback: str = ''):
        # If this reasoner has been configured to *not* accept planner
        # structured input, treat the incoming `task_description` as a normal
        # free-text task prompt and produce a single draft using the LLM.
        if not getattr(self, 'use_planning_input', True):
            prompt = (
                "You are asked to produce a draft for the following task.\n"
                "Format your output as a template-filled string.\n\n" + str(task_description)
            )
            messages = [{"role": "user", "content": prompt}]
            try:
                draft = self.llm(messages=messages, temperature=0.1)
            except Exception:
                draft = ''
            return [{
                'description': 'full_task',
                'tool_use_instruction': '',
                'draft': draft,
                'format': ''
            }]

        # task_description may be a JSON string or already a list
        plan = None
        if isinstance(task_description, (list, tuple)):
            plan = list(task_description)
        else:
            try:
                plan = json.loads(task_description)
            except Exception:
                # If parsing fails, fall back to a conservative behavior:
                # return a single-draft list using the raw text as the prompt.
                messages = [{"role": "user", "content": str(task_description)}]
                try:
                    draft = self.llm(messages=messages, temperature=0.1)
                except Exception:
                    draft = ''
                return [{
                    'description': 'fallback',
                    'tool_use_instruction': '',
                    'draft': draft,
                    'format': ''
                }]

        results = []
        for step in plan:
            # tolerate missing keys with safe defaults
            description = step.get('description', '')
            reasoning_instruction = step.get('reasoning_instruction', '')
            tool_use_instruction = step.get('tool_use_instruction', '')
            draft_template = step.get('draft_template', '')
            fmt = step.get('format', '')

            # fetch memory/context for this step (best-effort)
            memory_ctx = ''
            try:
                # memory modules expose a callable that returns text when used
                memory_ctx = self.memory(description) if self.memory else ''
            except Exception:
                try:
                    # some memory implementations may provide retriveMemory
                    memory_ctx = self.memory.retriveMemory(description)
                except Exception:
                    memory_ctx = ''

            prompt = (
                "You are asked to produce a filled drafting template.\n"
                "Draft template (fill exactly, keep format):\n" + draft_template + "\n\n"
                "Reasoning instructions:\n" + reasoning_instruction + "\n\n"
                "Relevant memories/context:\n" + (memory_ctx or "(none)") + "\n\n"
                "Produce ONLY the filled template content (no extra commentary)."
            )

            messages = [{"role": "user", "content": prompt}]
            try:
                draft = self.llm(messages=messages, temperature=0.1)
            except Exception:
                draft = ''

            results.append({
                'description': description,
                'tool_use_instruction': tool_use_instruction,
                'draft': draft,
                'format': fmt,
            })

        return results
    

