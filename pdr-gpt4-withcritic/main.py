import os
import openai
import csv
import time

from simulate_participant import Participant
from tasks import get_all_tasks
from evaluator import Evaluator
from baseline_adhoc import BaselineAdHocSimulator
from pdr_simulator import PDRSimulator
from pdr_simulator_critic import PDRSimulatorWithCritic
from critic import LLMCritic
from expert_evaluator import ExpertEvaluator  
from analysis import ExperimentAnalyzer

def save_results_to_csv(results, filename):
    """
    Saves a list of result dictionaries (each from a simulator run)
    to a CSV file. The keys of the first result item determine the CSV columns.
    """
    if not results:
        print(f"No results to save to {filename}.")
        return

    keys = results[0].keys()  # use the dict keys of the first item for the header

    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {filename}.")

def main():
    
    dir = os.path.dirname(__file__)
    with open(os.path.join(dir, "api_key"), "r") as f:
        openai.api_key = f.read().strip()

    # # 1) Create participants
    # participants = [
    #     Participant(
    #         name="Participant_A",
    #         persona_description="You are an expert in business writing with minimal creative flair."
    #     ),
    #     Participant(
    #         name="Participant_B",
    #         persona_description="You are a novice in technical writing, but love storytelling."
    #     ),
    #     Participant(
    #         name="Participant_C",
    #         persona_description="You are a seasoned educator focusing on concise, explanatory writing."
    #     ),
    #     Participant(
    #         name="Participant_D",
    #         persona_description="You are an AI enthusiast who loves imaginative and futuristic details."
    #     ),
    #     Participant(
    #         name="Participant_E",
    #         persona_description="You are a creative writer who frequently uses poetic language and metaphors."
    #     )
    # ]

    # # 2) Load tasks
    # tasks = get_all_tasks()

    # # 3) Create evaluator
    # evaluator = Evaluator(use_gpt4_for_eval=True, model="gpt-4o")

    # # 4) Create an optional expert evaluator (e.g., GPT-4o in a domain-expert role)
    # expert_evaluator = ExpertEvaluator(model="gpt-4o", temperature=0)


    # # 5) Set up simulators
    # # 5a) Baseline Ad Hoc
    # adhoc_simulator = BaselineAdHocSimulator(
    #     evaluator=evaluator,
    #     max_iterations=5,
    #     score_threshold=85,
    #     expert_evaluator=expert_evaluator
    # )
    # # 5b) PDR without Critic
    # pdr_simulator = PDRSimulator(
    #     evaluator=evaluator,
    #     max_iterations=5,
    #     score_threshold=85,
    #     num_outputs_per_iter=3
    # )
    # # 5c) PDR with Critic
    # custom_critic = LLMCritic(model="gpt-4o", temperature=0)
    # pdr_critic_simulator = PDRSimulatorWithCritic(
    #     evaluator=evaluator,
    #     max_iterations=5,
    #     score_threshold=85,
    #     num_outputs_per_iter=3,
    #     critic=custom_critic
    # )

    # # 6) Lists to collect results
    # all_results_adhoc = []
    # all_results_pdr = []
    # all_results_pdr_critic = []

    # # 7) Run simulations
    # for participant in participants:
    #     # print(f"\n=== {participant.name}: Baseline Ad Hoc ===")
    #     # for task in tasks:
    #     #     result = adhoc_simulator.simulate(participant, task)
    #     #     print(f"Task: {task.name}, Score: {result['final_score']}, ")
    #     #     print(f"iteration_count: {result['iteration_count']}")
    #     #     print(f"Time Spent: {result['time_spent_sec']:.2f} sec\n")
    #     #     all_results_adhoc.append(result)

    #     # print(f"\n=== {participant.name}: PDR (No Critic) ===")
    #     # for task in tasks:
    #     #     result = pdr_simulator.simulate(participant, task)
    #     #     print(f"Task: {task.name}, Score: {result['final_score']}, ")
    #     #     print(f"iteration_count: {result['iteration_count']}")
    #     #     print(f"Time Spent: {result['time_spent_sec']:.2f} sec\n")
    #     #     all_results_pdr.append(result)

    #     print(f"\n=== {participant.name}: PDR WITH Critic ===")
    #     for task in tasks:
    #         result = pdr_critic_simulator.simulate(participant, task)
    #         print(f"Task: {task.name}, Score: {result['final_score']}, ")
    #         print(f"iteration_count: {result['iteration_count']}")
    #         print(f"Time Spent: {result['time_spent_sec']:.2f} sec\n")
    #         all_results_pdr_critic.append(result)

    # # 7) Save results to CSV
    # timestamp = int(time.time())
    # adhoc_file = f"adhoc_results_{timestamp}.csv"
    # pdr_file = f"pdr_results_{timestamp}.csv"
    # pdr_critic_file = f"pdr_critic_results_{timestamp}.csv"

    # # save_results_to_csv(all_results_adhoc, adhoc_file)
    # # save_results_to_csv(all_results_pdr, pdr_file)
    # save_results_to_csv(all_results_pdr_critic, pdr_critic_file)


    # -------------------------------------------------------------------------
    # ANALYSIS (Descriptive & Inferential)
    # -------------------------------------------------------------------------
    analyzer = ExperimentAnalyzer()

    adhoc_file = "D://vanderbilt//chatGPT//4_preference_driven_refinement//pdr-gpt4-withcritic//adhoc_results_1738695949.csv"
    pdr_file = "D://vanderbilt//chatGPT//4_preference_driven_refinement//pdr-gpt4-withcritic//pdr_results_1738695949.csv"
    pdr_critic_file = "D://vanderbilt//chatGPT//4_preference_driven_refinement//pdr-gpt4-withcritic//pdr_critic_results_1751439618.csv"

    # 1) Load data from CSV for Ad Hoc and PDR
    adhoc_data = analyzer.load_results(adhoc_file)
    pdr_data = analyzer.load_results(pdr_file)
    pdr_critic_data = analyzer.load_results(pdr_critic_file)

    # 2) DESCRIPTIVE STATS: e.g., iteration_count, time_spent_sec
    for measure in ["iteration_count", "time_spent_sec", "final_score"]:
        adhoc_stats = analyzer.descriptive_stats(adhoc_data, measure)
        pdr_stats = analyzer.descriptive_stats(pdr_data, measure)
        pdr_critic_stats = analyzer.descriptive_stats(pdr_critic_data, measure)

        print(f"\n[Descriptive Stats for {measure}]")
        print(f"Ad Hoc => mean: {adhoc_stats['mean']:.2f}, stdev: {adhoc_stats['stdev']:.2f}, count: {adhoc_stats['count']}")
        print(f"PDR    => mean: {pdr_stats['mean']:.2f}, stdev: {pdr_stats['stdev']:.2f}, count: {pdr_stats['count']}")
        print(f"PDR Critic   => mean: {pdr_critic_stats['mean']:.2f}, stdev: {pdr_critic_stats['stdev']:.2f}, count: {pdr_critic_stats['count']}")

    # 3) INFERENTIAL STATS: e.g., Paired T-test comparing Ad Hoc vs. PDR on iteration_count
    measure = "iteration_count"
    result = analyzer.compare_two_conditions(adhoc_data, pdr_data, measure=measure, test_type="paired_t")
    if "error" not in result:
        print(f"\n[Inferential Stats for {measure} (Ad Hoc vs. PDR)]")
        print(f"Test: {result['test']}")
        print(f"Statistic: {result['statistic']:.3f}, p-value: {result['p_value']:.3f}")
    else:
        print(f"\nError in statistical test for {measure}: {result['error']}")

if __name__ == "__main__":
    main()
