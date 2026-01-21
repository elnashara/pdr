import os
import openai
import csv
import time

from simulate_participant import Participant
from tasks import get_all_tasks
from evaluator import Evaluator
from adhoc_simulator import AdHocSimulator
from pdr_simulator_non_critic import PDRSimulatorNonCritic
from pdr_simulator_critic import PDRSimulatorCritic
from critic import LLMCritic
from expert_evaluator import ExpertEvaluator  
from analysis import ExperimentAnalyzer

from pathlib import Path

def append_dicts_to_csv(rows, filepath):
    """
    Append one or more dict rows to a CSV.
    - Creates the file (and parent folder) if missing.
    - Writes header once (on first creation).
    - Uses existing header thereafter; extra keys are ignored; missing keys become empty cells.
    """
    if not rows:
        return

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    file_exists = os.path.exists(filepath) and os.path.getsize(filepath) > 0
    header = None

    if file_exists:
        # Read existing header to preserve column order
        with open(filepath, "r", newline="", encoding="utf-8") as rf:
            reader = csv.reader(rf)
            header = next(reader, None)
            if not header:
                file_exists = False  # treat as new file if header missing

    if not file_exists:
        # First write: compute header from provided rows (first-seen key order)
        seen = set()
        header = []
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    header.append(k)
        with open(filepath, "w", newline="", encoding="utf-8") as wf:
            writer = csv.DictWriter(wf, fieldnames=header, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return

    # Append using existing header
    with open(filepath, "a", newline="", encoding="utf-8") as af:
        writer = csv.DictWriter(af, fieldnames=header, extrasaction="ignore")
        for r in rows:
            writer.writerow(r)

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

    # # # 1) Create participants
    # # participants = [
    # #     Participant(
    # #         name="Participant_A",
    # #         persona_description=(
    # #             "You are a senior backend engineer. You write highly structured, "
    # #             "maintainable code with thorough test coverage. You always focus on "
    # #             "edge cases and correctness but sometimes sacrifice brevity."
    # #         )
    # #     ),
    # #     Participant(
    # #         name="Participant_B",
    # #         persona_description=(
    # #             "You are a junior developer. You over-explain in comments, mix styles, "
    # #             "and sometimes miss required technical keywords, but your enthusiasm "
    # #             "leads to creative trial-and-error approaches."
    # #         )
    # #     ),
    # #     Participant(
    # #         name="Participant_C",
    # #         persona_description=(
    # #             "You are a software architect and educator. You emphasize clarity, "
    # #             "clean separation of concerns, and design principles like SRP. "
    # #             "Your outputs are explanatory and well-structured, sometimes verbose."
    # #         )
    # #     ),
    # #     Participant(
    # #         name="Participant_D",
    # #         persona_description=(
    # #             "You are an open-source contributor and AI enthusiast. You like to "
    # #             "experiment with speculative or futuristic approaches. Your solutions "
    # #             "often include imaginative extensions that may not strictly meet rubric requirements."
    # #         )
    # #     ),
    # #     Participant(
    # #         name="Participant_E",
    # #         persona_description=(
    # #             "You are a documentation-focused developer. You write detailed commit "
    # #             "messages, API docs, and bug reports with a polished narrative style. "
    # #             "Your code explanations are clear but sometimes too wordy."
    # #         )
    # #     ),
    # #     Participant(
    # #         name="Participant_F",
    # #         persona_description=(
    # #             "You are a QA engineer. You prioritize correctness, validation, and "
    # #             "complete coverage. You often critique missing cases and ambiguous "
    # #             "statements, but sometimes your outputs are overly strict and rigid."
    # #         )
    # #     ),
    # #     Participant(
    # #         name="Participant_G",
    # #         persona_description=(
    # #             "You are a DevOps/SRE engineer. You focus on automation, deployment, "
    # #             "and monitoring concerns. You often emphasize scalability and resilience, "
    # #             "but may overlook low-level implementation details."
    # #         )
    # #     ),
    # #     Participant(
    # #         name="Participant_H",
    # #         persona_description=(
    # #             "You are a product manager with technical writing skills. You frame "
    # #             "outputs around end-user value, business impact, and clarity. Your "
    # #             "summaries are persuasive but may lack detailed technical depth."
    # #         )
    # #     )
    # # ]

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
    # evaluator = Evaluator(use_gpt5_for_eval=True, model="gpt-5")

    # # 4) Create an optional expert evaluator (e.g., GPT-5 in a domain-expert role)
    # expert_evaluator = ExpertEvaluator(model="gpt-5", temperature=0)

    # # 5) Set up simulators
    # # 5a) Baseline Ad Hoc
    # adhoc_simulator = AdHocSimulator(
    #     evaluator=evaluator,
    #     max_iterations=5,
    #     score_threshold=85,
    #     expert_evaluator=expert_evaluator
    # )
    # # 5b) PDR without Critic
    # pdr_simulator = PDRSimulatorNonCritic(
    #     evaluator=evaluator,
    #     max_iterations=5,
    #     score_threshold=85,
    #     num_outputs_per_iter=3
    # )
    # # 5c) PDR with Critic
    # custom_critic = LLMCritic(model="gpt-5")
    # pdr_critic_simulator = PDRSimulatorCritic(
    #     evaluator=evaluator,
    #     max_iterations=5,
    #     score_threshold=85,
    #     num_outputs_per_iter=3,
    #     critic=custom_critic
    # )

   
    # # One timestamp per run so all appends go to the same file
    # timestamp = int(time.time())

    # # (Optional) keep files in a results/ folder
    # results_dir = "results"
    # adhoc_file = os.path.join(results_dir, f"adhoc_results_{timestamp}.csv")
    # pdr_file = os.path.join(results_dir, f"pdr_results_{timestamp}.csv")
    # pdr_critic_file = os.path.join(results_dir, f"pdr_critic_results_{timestamp}.csv")

    # # 6) Lists to collect results
    # all_results_adhoc = []
    # all_results_pdr = []
    # all_results_pdr_critic = []

    # # Run simulations
    # for participant in participants:
    #     print(f"\n=== {participant.name}: Baseline Ad Hoc ===")
    #     for task in tasks:
    #         result = adhoc_simulator.simulate(participant, task)
    #         print(f"Task: {task.name}, Score: {result['final_score']}, ")
    #         print(f"iteration_count: {result['iteration_count']}")
    #         print(f"Time Spent: {result['time_spent_sec']:.2f} sec\n")
    #         all_results_adhoc.append(result)
    #         append_dicts_to_csv([result], adhoc_file)   # <-- append per inner loop
    #     print("\n===========================================")

    # for participant in participants:
    #     print(f"\n=== {participant.name}: PDR (No Critic) ===")
    #     for task in tasks:
    #         result = pdr_simulator.simulate(participant, task)
    #         print(f"Task: {task.name}, Score: {result['final_score']}, ")
    #         print(f"iteration_count: {result['iteration_count']}")
    #         print(f"Time Spent: {result['time_spent_sec']:.2f} sec\n")
    #         all_results_pdr.append(result)
    #         append_dicts_to_csv([result], pdr_file)     # <-- append per inner loop
    #     print("\n===========================================")

    # for participant in participants:
    #     print(f"\n=== {participant.name}: PDR WITH Critic ===")
    #     for task in tasks:
    #         result = pdr_critic_simulator.simulate(participant, task)
    #         print(f"Task: {task.name}, Score: {result['final_score']}, ")
    #         print(f"iteration_count: {result['iteration_count']}")
    #         print(f"Time Spent: {result['time_spent_sec']:.2f} sec\n")
    #         all_results_pdr_critic.append(result)
    #         append_dicts_to_csv([result], pdr_critic_file)  # <-- append per inner loop
    #     print("\n===========================================")

    # # You can still write aggregate CSVs at the end if you want separate “all_*” files
    # end_ts = int(time.time())
    # append_dicts_to_csv(all_results_adhoc, os.path.join(results_dir, f"adhoc_results_all_{end_ts}.csv"))
    # append_dicts_to_csv(all_results_pdr, os.path.join(results_dir, f"pdr_results_all_{end_ts}.csv"))
    # append_dicts_to_csv(all_results_pdr_critic, os.path.join(results_dir, f"pdr_critic_results_all_{end_ts}.csv"))



    # -------------------------------------------------------------------------
    # ANALYSIS (Descriptive & Inferential)
    # -------------------------------------------------------------------------
    analyzer = ExperimentAnalyzer()

    adhoc_file = "D://vanderbilt//chatGPT//4_preference_driven_refinement//pdr-gpt5//results//adhoc_results_1758563132.csv"
    pdr_file = "D://vanderbilt//chatGPT//4_preference_driven_refinement//pdr-gpt5//results//pdr_results_1758563132.csv"
    pdr_critic_file = "D://vanderbilt//chatGPT//4_preference_driven_refinement//pdr-gpt5//results//pdr_critic_results_1758563132.csv"

    # 1) Load data from CSV for Ad Hoc and PDR
    adhoc_data = analyzer.load_results(adhoc_file)
    pdr_data = analyzer.load_results(pdr_file)
    pdr_critic_data = analyzer.load_results(pdr_critic_file)  # if you want to compare that too

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
    # Ad Hoc vs PDR
    res_adhoc_vs_pdr = analyzer.compare_two_conditions(
        adhoc_data, pdr_data, measure=measure, test_type="paired_t"
    )

    if "error" not in res_adhoc_vs_pdr:
        print(f"\n[Inferential Stats for {measure} (Ad Hoc vs. PDR)]")
        print(f"Test: {res_adhoc_vs_pdr['test']}")
        print(f"Statistic: {res_adhoc_vs_pdr['statistic']:.3f}, p-value: {res_adhoc_vs_pdr['p_value']:.3f}")
    else:
        print(f"\nError in statistical test for {measure}: {res_adhoc_vs_pdr['error']}")

    # PDR vs PDR+Critic (rename your var to pdr_critic_data, not *_stats)
    res_pdr_vs_critic = analyzer.compare_two_conditions(
        pdr_data, pdr_critic_data, measure=measure, test_type="paired_t"
    )
    if "error" not in res_pdr_vs_critic:
        print(f"\n[Inferential Stats for {measure} (PDR vs PDR+Critic)]")
        print(f"Test: {res_pdr_vs_critic['test']}")
        print(f"Statistic: {res_pdr_vs_critic['statistic']:.3f}, p-value: {res_pdr_vs_critic['p_value']:.3f}")
    else:
        print(f"\nError in statistical test for {measure}: {res_pdr_vs_critic['error']}")

if __name__ == "__main__":
    main()
