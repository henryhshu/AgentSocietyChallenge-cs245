from dotenv import load_dotenv
import os
from websocietysimulator import Simulator
from websocietysimulator.agent import SimulationAgent
import json 
from websocietysimulator.llm import LLMBase, OpenAILLM
from websocietysimulator.agent.modules.planning_modules import PlanningBase 
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from websocietysimulator.agent.modules.memory_modules import MemoryDILU
import logging
logging.basicConfig(level=logging.INFO)

class PlanningBaseline(PlanningBase):
    """Inherit from PlanningBase"""
    
    def __init__(self, llm):
        """Initialize the planning module"""
        super().__init__(llm=llm)
    
    def __call__(self, task_description):
        """Override the parent class's __call__ method"""
        self.plan = [
            {
                'description': 'First I need to find user information',
                'reasoning instruction': 'None', 
                'tool use instruction': {task_description['user_id']}
            },
            {
                'description': 'Next, I need to find item information',
                'reasoning instruction': 'None',
                'tool use instruction': {task_description['item_id']}
            }
        ]
        return self.plan


class ReasoningBaseline(ReasoningBase):
    """Inherit from ReasoningBase"""
    
    def __init__(self, profile_type_prompt, llm):
        """Initialize the reasoning module"""
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)
        
    def __call__(self, task_description: str):
        """Override the parent class's __call__ method"""
        prompt = '''
{task_description}'''
        prompt = prompt.format(task_description=task_description)
        
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.0,
            max_tokens=1000
        )
        
        return reasoning_result

class ReasoningVoteRating(ReasoningBase):
    """Inherit from ReasoningBase"""
    
    def __init__(self, profile_type_prompt, llm):
        """Initialize the reasoning module"""
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)
        
    def __call__(self, task_description: str):
        """Override the parent class's __call__ method"""
        prompt = '''
{task_description}'''
        prompt = prompt.format(task_description=task_description)

        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            n=3
        )

        vote_results = [set() for _ in range(5)]
        for vote_result in reasoning_result:
            try:
                stars_line = [line for line in vote_result.split('\n') if 'stars:' in line][0]
                stars = float(stars_line.split(':')[1].strip())
                if stars >= 1.0 and stars <= 5.0:
                    vote_results[int(stars)-1].add(vote_result)
            except:
                continue
        
        final_set = max(vote_results, key=lambda x: len(x))
        final_result = final_set.pop() if len(final_set) > 0 else reasoning_result[0]
        return final_result


class MySimulationAgent(SimulationAgent):
    """Participant's implementation of SimulationAgent."""
    
    def __init__(self, llm: LLMBase):
        """Initialize MySimulationAgent"""
        super().__init__(llm=llm)
        self.planning = PlanningBaseline(llm=self.llm)
        # self.reasoning = ReasoningBaseline(profile_type_prompt='', llm=self.llm)
        self.reasoning = ReasoningVoteRating(profile_type_prompt='', llm=self.llm)
        self.memory = MemoryDILU(llm=self.llm)
        
    def workflow(self):
        """
        Simulate user behavior
        Returns:
            tuple: (star (float), useful (float), funny (float), cool (float), review_text (str))
        """
        try:
            plan = self.planning(task_description=self.task)

            for sub_task in plan:
                if 'user' in sub_task['description']:
                    user = str(self.interaction_tool.get_user(user_id=self.task['user_id']))
                elif 'item' in sub_task['description']:
                    item = self.interaction_tool.get_item(item_id=self.task['item_id'])
                    item.pop("images", None)  # Remove images to reduce context size
                    item.pop("videos", None)  # Remove videos to reduce context size
                    item = str(item)
            reviews_item = self.interaction_tool.get_reviews(item_id=self.task['item_id'])
            for review in reviews_item:
                review_text = review['title'] + "\n" +review['text']
                self.memory(f'review: {review_text}')
            reviews_user = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
            review_similar = self.memory(f'{reviews_user[0]["text"]}')
            task_description = f'''
            You are an Amazon user. Here is your Amazon profile and review history: {user}

            Write a review for this item: {item}

            Others have reviewed this item before: {review_similar}

            Format your response exactly as follows:
            stars: [your rating]
            review: [your review]
            '''
            result = self.reasoning(task_description)
            
            try:
                stars_line = [line for line in result.split('\n') if 'stars:' in line][0]
                review_line = [line for line in result.split('\n') if 'review:' in line][0]
            except:
                print('Error:', result)

            stars = float(stars_line.split(':')[1].strip())
            review_text = review_line.split(':')[1].strip()

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
    # Load env variables
    load_dotenv()
    data_dir = os.getenv("DATA_DIR")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Set the data
    task_set = "amazon" # "goodreads" or "yelp"
    simulator = Simulator(data_dir=data_dir, device="auto", cache=True)
    simulator.set_task_and_groundtruth(task_dir=f"./track1/{task_set}/tasks", groundtruth_dir=f"./track1/{task_set}/groundtruth")

    # Set the agent and LLM
    simulator.set_agent(MySimulationAgent)
    simulator.set_llm(OpenAILLM(api_key=openai_api_key))

    # Run the simulation
    # If you don't set the number of tasks, the simulator will run all tasks.
    outputs = simulator.run_simulation(number_of_tasks=200, enable_threading=True, max_workers=10)
    
    # Evaluate the agent
    evaluation_results = simulator.evaluate()       
    with open(f'./evaluation_results_track1_{task_set}.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    # Get evaluation history
    evaluation_history = simulator.get_evaluation_history()