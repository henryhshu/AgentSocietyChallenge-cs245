from websocietysimulator import Simulator
from websocietysimulator.agent import SimulationAgent
import json 
import os
from websocietysimulator.llm import LLMBase, GeminiLLM
from websocietysimulator.agent.modules.planning_modules import PlanningBase, PlanningTemplateDrafting, PlanningVoyager, PlanningDEPS, PlanningIO, PlanningTD
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase, ReasoningTemplateDrafting, ReasoningDILU, ReasoningSelfRefine
from websocietysimulator.agent.modules.memory_modules import MemoryHybrid, MemoryDILU, MemoryVoyager, MemoryGenerative, MemoryTP, MemoryIngrid
import logging
from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO)

class PlanningBaseline(PlanningBase):
    """Inherit from PlanningBase"""
    
    def __init__(self, llm):
        """Initialize the planning module"""
        super().__init__(llm=llm)
    
    def __call__(self, *args, **kwargs):
        """Override the parent class's __call__ method.

        Supports two calling conventions for backward compatibility:
        1) legacy: __call__(task_description_dict)
        2) new: __call__(task_type, task_description, feedback, few_shot)
        """
        # Determine which form was used and extract task_description dict
        task_description = None
        if len(args) == 1:
            task_description = args[0]
        elif len(args) >= 2:
            # (task_type, task_description, ...)
            task_description = args[1]
        else:
            task_description = kwargs.get('task_description')

        # If task_description is a JSON string, try to decode it
        if isinstance(task_description, str):
            try:
                import json as _json
                task_description = _json.loads(task_description)
            except Exception:
                # leave as string if it isn't JSON
                pass

        # Ensure we have a dict-like task_description to read ids from
        if not isinstance(task_description, dict):
            raise ValueError("PlanningBaseline expected a task_description dict (or JSON string). Got: %r" % (task_description,))

        self.plan = [
            {
                'description': 'First I need to find user information',
                'reasoning instruction': 'None', 
                'tool use instruction': {task_description['user_id']}
            },
            {
                'description': 'Next, I need to find business information',
                'reasoning instruction': 'None',
                'tool use instruction': {task_description['item_id']}
            }
        ]
        return self.plan


class ReasoningBaseline(ReasoningBase):
    """Inherit from ReasoningBase"""
    
    def __init__(self, profile_type_prompt, llm, use_planning_input: bool = True):
        """Initialize the reasoning module

        Args:
            profile_type_prompt: prompt guiding profile-specific behavior
            llm: LLM instance
            use_planning_input: whether to accept planner output as structured input
        """
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm, use_planning_input=use_planning_input)
        
    def __call__(self, task_description: str, plan=None):
        """Override the parent class's __call__ method. Accept an optional
        `plan` produced by the planner and include it in the prompt."""
        def _stringify_plan(plan_obj):
            try:
                import json as _json
                return _json.dumps(plan_obj)
            except Exception:
                return str(plan_obj)

        # support receiving an optional plan (list/dict) from the planner
        plan_text = ''
        if plan is not None:
            plan_text = _stringify_plan(plan)

        if plan_text:
            prompt = f"\nPlan:\n{plan_text}\n\n{task_description}"
        else:
            prompt = f"\n{task_description}"
        
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.0,
            max_tokens=1000
        )
        
        return reasoning_result


class SimulationAgentWorkflow(SimulationAgent):
    """Participant's implementation of SimulationAgent."""
    
    def __init__(self, llm: LLMBase):
        """Initialize MySimulationAgent"""
        super().__init__(llm=llm)
        # Use the new template-based drafting planner to produce strict plan templates
        #self.planning = PlanningTemplateDrafting(llm=self.llm)
        self.memory = MemoryDILU(llm=self.llm)
        self.planning = PlanningIO(llm=self.llm)
        # Allow toggling whether the reasoner consumes planner output using
        # environment variable `ENABLE_PLANNING_INPUT` (default: enabled).
        self.reasoning = ReasoningBaseline(profile_type_prompt='', llm=self.llm, use_planning_input=False)
    
    def workflow(self):
        """
        Simulate user behavior
        Returns:
            tuple: (star (float), useful (float), funny (float), cool (float), review_text (str))
        """
        try:
            # Planner expects: task_type, task_description, feedback, few_shot
            # Serialize the task dict to a JSON string so the planner receives a stable representation
            plan = self.planning('review', json.dumps(self.task), '', '')

            # initialize variables in case the planner does not include the
            # expected subtasks (avoids UnboundLocalError when using them later)
            user = None
            business = None
            for sub_task in plan:
                desc = str(sub_task.get('description', '')).lower() if isinstance(sub_task, dict) else str(sub_task).lower()
                if 'user' in desc:
                    user = str(self.interaction_tool.get_user(user_id=self.task['user_id']))
                elif 'business' in desc or 'item' in desc:
                    business = str(self.interaction_tool.get_item(item_id=self.task['item_id']))

            # Fallbacks: if planner didn't provide subtask that identifies user/business,
            # fetch them directly so downstream code can continue.
            if user is None:
                user = str(self.interaction_tool.get_user(user_id=self.task['user_id']))
            if business is None:
                business = str(self.interaction_tool.get_item(item_id=self.task['item_id']))
            reviews_item = self.interaction_tool.get_reviews(item_id=self.task['item_id'])
            for review in reviews_item:
                review_text = review['text']
                self.memory(f'review: {review_text}')
            reviews_user = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
            review_similar = self.memory(f'{reviews_user[0]["text"]}')
            task_description = f'''
            You are a real human user on Yelp, a platform for crowd-sourced business reviews. Here is your Yelp profile and review history: {user}

            You need to write a review for this business: {business}

            Others have reviewed this business before: {review_similar}

            Please analyze the following aspects carefully:
            1. Based on your user profile and review style, what rating would you give this business? Remember that many users give 5-star ratings for excellent experiences that exceed expectations, and 1-star ratings for very poor experiences that fail to meet basic standards.
            2. Given the business details and your past experiences, what specific aspects would you comment on? Focus on the positive aspects that make this business stand out or negative aspects that severely impact the experience.
            3. Consider how other users might engage with your review in terms of:
            - Useful: How informative and helpful is your review?
            - Funny: Does your review have any humorous or entertaining elements?
            - Cool: Is your review particularly insightful or praiseworthy?

            Requirements:
            - Star rating must be one of: 1.0, 2.0, 3.0, 4.0, 5.0
            - If the business meets or exceeds expectations in key areas, consider giving a 5-star rating
            - If the business fails significantly in key areas, consider giving a 1-star rating
            - Review text should be 2-4 sentences, focusing on your personal experience and emotional response
            - Useful/funny/cool counts should be non-negative integers that reflect likely user engagement
            - Maintain consistency with your historical review style and rating patterns
            - Focus on specific details about the business rather than generic comments
            - Be generous with ratings when businesses deliver quality service and products
            - Be critical when businesses fail to meet basic standards

            Format your response exactly as follows:
            stars: [your rating]
            review: [your review]
            '''
            # task_description = f'''
            # You are a Yelp user. Here is your Yelp profile and review history: {user}

            # Write a review for this business: {business}

            # Others have reviewed this business before: {review_similar}

            # Format your response exactly as follows:
            # stars: [your rating]
            # review: [2-4 sentence review, focus on personal experience]
            # '''
            
            # The reasoner may return different shapes depending on which
            # Reasoning class is used: a plain string, a list of draft dicts,
            # or a single dict. Normalize to a single string `chosen_draft`.
            raw = self.reasoning(task_description, plan)

            chosen_draft = ''
            # If reasoner returned a list of dicts (ReasoningTemplateDrafting),
            # pick the first draft that looks like it contains stars+review.
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, dict):
                        dt = item.get('draft', '')
                        if isinstance(dt, str) and 'stars' in dt.lower() and 'review' in dt.lower():
                            chosen_draft = dt
                            break
                if not chosen_draft and len(raw) > 0:
                    first = raw[0]
                    chosen_draft = first.get('draft', '') if isinstance(first, dict) else str(first)
            elif isinstance(raw, dict):
                # single dict output — try to extract a 'draft' field
                chosen_draft = raw.get('draft', '') if raw else ''
            else:
                # assume string-like
                chosen_draft = str(raw)

            # Defensive parsing: extract stars and review using regex. This
            # avoids UnboundLocalError when expected lines are missing.
            # Robust parsing pipeline for the filled draft. Try multiple
            # strategies so the workflow tolerates different model outputs.
            try:
                import re
                import json as _json

                def word_to_number(w: str):
                    mapping = {
                        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                        '1': 1, '2': 2, '3': 3, '4': 4, '5': 5
                    }
                    return mapping.get(w.lower())

                parsed = False

                # 1) If the draft is JSON (or contains a JSON object), try to load it.
                try:
                    maybe = chosen_draft.strip()
                    if maybe.startswith('{') or '"stars"' in maybe or "'stars'" in maybe:
                        j = _json.loads(maybe)
                        if isinstance(j, dict) and 'stars' in j:
                            stars = float(j.get('stars') or 0)
                            review_text = str(j.get('review') or j.get('text') or '')
                            parsed = True
                except Exception:
                    parsed = False

                # 2) Regex for explicit 'stars: X' and 'review: ...'
                if not parsed:
                    stars_m = re.search(r'stars\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)', chosen_draft, re.IGNORECASE)
                    review_m = re.search(r'review\s*[:\-]?\s*(.*)', chosen_draft, re.IGNORECASE | re.DOTALL)
                    if stars_m:
                        stars = float(stars_m.group(1))
                        parsed = True
                    if review_m:
                        review_text = review_m.group(1).strip()
                        parsed = True if parsed else parsed

                # 3) 'X star' patterns or 'rated X stars'
                if not parsed:
                    star_phrase = re.search(r'([0-9])\s*-?\s*star', chosen_draft, re.IGNORECASE)
                    if star_phrase:
                        stars = float(star_phrase.group(1))
                        parsed = True
                    else:
                        words_star = re.search(r'(one|two|three|four|five)\s+star', chosen_draft, re.IGNORECASE)
                        if words_star:
                            val = word_to_number(words_star.group(1))
                            if val:
                                stars = float(val)
                                parsed = True

                # 4) Try to find any standalone digit that looks like a rating (1-5)
                if not parsed:
                    any_digit = re.search(r'\b([1-5])\b', chosen_draft)
                    if any_digit:
                        stars = float(any_digit.group(1))
                        parsed = True

                # 5) As a last resort, ask the LLM to extract structured fields
                if not parsed:
                    try:
                        # Use a short parsing prompt to the same LLM wrapper
                        parse_prompt = (
                            "Extract and return a JSON object with keys 'stars' (int 1-5)"
                            ", and 'review' (string). If not found, use 0 for stars. "
                            "Input:\n" + chosen_draft + "\n\nJSON:")
                        resp = self.llm(messages=[{"role": "user", "content": parse_prompt}], temperature=0.0)
                        j = _json.loads(resp)
                        if isinstance(j, dict) and 'stars' in j:
                            stars = float(j.get('stars') or 0)
                            review_text = str(j.get('review') or '')
                            parsed = True
                    except Exception:
                        parsed = False

                if not parsed:
                    print('Warning: could not parse stars from draft; defaulting to 0.0. Draft:')
                    print(chosen_draft)
                    stars = 0.0
            except Exception as e:
                print('Error parsing draft:', e, repr(chosen_draft))
                stars = 0.0
                review_text = ''

            if len(review_text) > 512:
                review_text = review_text[:512]
                
            return {
                "stars": stars,
                "review": review_text
            }
        except Exception as e:
            print(f"Error in workflow: {e}")
            return {
                "stars": 0,
                "review": ""
            }

if __name__ == "__main__":
    # Set the data
    task_set = "yelp" #"amazon" # "goodreads" or "yelp"
    simulator = Simulator(data_dir="data", device="gpu", cache=True)
    simulator.set_task_and_groundtruth(task_dir=f"example/track1/{task_set}/tasks", groundtruth_dir=f"example/track1/{task_set}/groundtruth")

    # Set the agent and LLM (use Google Gemini)
    simulator.set_agent(SimulationAgentWorkflow)
    
    api_key = os.getenv("GEMINI_API_KEY")
    simulator.set_llm(GeminiLLM(api_key=api_key, model="gemini-2.0-flash"))

    # Run the simulation
    # If you don't set the number of tasks, the simulator will run all tasks.
    outputs = simulator.run_simulation(number_of_tasks=50, enable_threading=True, max_workers=10)
    
    # Evaluate the agent
    evaluation_results = simulator.evaluate()
    with open(f'./report/evaluation_results_planningio_{task_set}.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    # Get evaluation history
    evaluation_history = simulator.get_evaluation_history()