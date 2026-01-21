import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import ace_tools as tools

# Load the CSV file
file_path = "results_1738695949.csv"
df = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
df.head()


# Group by 'method' and calculate mean and standard deviation for 'final_score'
quality_stats = df.groupby('method')['final_score'].agg(['mean', 'std'])

# Extract values for plotting
pdr_mean = quality_stats.loc['pdr', 'mean']
pdr_std = quality_stats.loc['pdr', 'std']
adhoc_mean = quality_stats.loc['adhoc', 'mean']
adhoc_std = quality_stats.loc['adhoc', 'std']

# Display calculated values
pdr_mean, pdr_std, adhoc_mean, adhoc_std



import matplotlib.pyplot as plt

# # Calculate mean and standard deviation for both methods
# pdr_mean = 4.5
# pdr_std = 0.3
# adhoc_mean = 3.8
# adhoc_std = 0.6





# Methods and corresponding values
methods = ['PDR-Generated', 'Ad Hoc-Generated']
means = [pdr_mean, adhoc_mean]
std_devs = [pdr_std, adhoc_std]

# Create bar chart with error bars
plt.figure(figsize=(8, 6))
plt.bar(methods, means, yerr=std_devs, capsize=10, color=['blue', 'orange'])
plt.xlabel('Refinement Method')
plt.ylabel('Mean Quality Rating')
plt.title('Comparison of Output Quality Ratings')
plt.ylim(3, 5)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Display the graph
plt.show()




# Group by method and calculate mean values
method_summary = df.groupby("method").agg(
    avg_final_score=("final_score", "mean"),
    avg_time_spent=("time_spent_sec", "mean"),
    avg_iteration_count=("iteration_count", "mean")
).reset_index()

# Define colors for consistency
colors = {"adhoc": "royalblue", "pdr": "darkorange"}

# Plot Average Final Score by Method with Colors
plt.figure(figsize=(8, 5))
plt.bar(method_summary["method"], method_summary["avg_final_score"], color=[colors[m] for m in method_summary["method"]])
plt.xlabel("Method")
plt.ylabel("Average Final Score")
plt.title("Average Final Score by Method")
plt.ylim(0, 100)
plt.show()

# Plot Average Time Spent by Method with Colors
plt.figure(figsize=(8, 5))
plt.bar(method_summary["method"], method_summary["avg_time_spent"], color=[colors[m] for m in method_summary["method"]])
plt.xlabel("Method")
plt.ylabel("Average Time Spent (Seconds)")
plt.title("Average Time Spent by Method")
plt.show()

# Plot Final Score Distribution with Colors
plt.figure(figsize=(8, 5))
for method in df["method"].unique():
    plt.hist(df[df["method"] == method]["final_score"], alpha=0.5, label=method, bins=10, color=colors[method])
plt.xlabel("Final Score")
plt.ylabel("Frequency")
plt.title("Final Score Distribution by Method")
plt.legend()
plt.show()

# Plot Average Iteration Count by Method with Colors
plt.figure(figsize=(8, 5))
plt.bar(method_summary["method"], method_summary["avg_iteration_count"], color=[colors[m] for m in method_summary["method"]])
plt.xlabel("Method")
plt.ylabel("Average Iteration Count")
plt.title("Average Iteration Count by Method")
plt.show()

# Group by method and participant to analyze performance variations
participant_summary = df.groupby(["method", "participant_name"]).agg(
    avg_final_score=("final_score", "mean"),
    avg_time_spent=("time_spent_sec", "mean"),
    avg_iteration_count=("iteration_count", "mean")
).reset_index()

# Plot Average Final Score by Participant for Each Method with Colors
plt.figure(figsize=(12, 6))
for method in df["method"].unique():
    subset = participant_summary[participant_summary["method"] == method]
    plt.bar(subset["participant_name"], subset["avg_final_score"], alpha=0.7, label=method, color=colors[method])

plt.xlabel("Participant Name")
plt.ylabel("Average Final Score")
plt.title("Average Final Score by Participant for Each Method")
plt.xticks(rotation=45)
plt.legend()
plt.show()

# Plot Average Time Spent by Participant for Each Method with Colors
plt.figure(figsize=(12, 6))
for method in df["method"].unique():
    subset = participant_summary[participant_summary["method"] == method]
    plt.bar(subset["participant_name"], subset["avg_time_spent"], alpha=0.7, label=method, color=colors[method])

plt.xlabel("Participant Name")
plt.ylabel("Average Time Spent (Seconds)")
plt.title("Average Time Spent by Participant for Each Method")
plt.xticks(rotation=45)
plt.legend()
plt.show()

# Plot Average Iteration Count by Participant for Each Method with Colors
plt.figure(figsize=(12, 6))
for method in df["method"].unique():
    subset = participant_summary[participant_summary["method"] == method]
    plt.bar(subset["participant_name"], subset["avg_iteration_count"], alpha=0.7, label=method, color=colors[method])

plt.xlabel("Participant Name")
plt.ylabel("Average Iteration Count")
plt.title("Average Iteration Count by Participant for Each Method")
plt.xticks(rotation=45)
plt.legend()
plt.show()

import seaborn as sns

# Scatter plot: Final Score vs. Time Spent, colored by Method
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="time_spent_sec", y="final_score", hue="method", palette=colors, alpha=0.7)
plt.xlabel("Time Spent (Seconds)")
plt.ylabel("Final Score")
plt.title("Final Score vs. Time Spent (Colored by Method)")
plt.legend(title="Method")
plt.show()

# Box plot: Distribution of Final Scores per Method with Colors
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="method", y="final_score", palette=colors)
plt.xlabel("Method")
plt.ylabel("Final Score")
plt.title("Distribution of Final Scores by Method")
plt.show()

# Violin plot: Iteration Count per Method with Colors
plt.figure(figsize=(8, 6))
sns.violinplot(data=df, x="method", y="iteration_count", inner="quartile", palette=colors)
plt.xlabel("Method")
plt.ylabel("Iteration Count")
plt.title("Iteration Count Distribution by Method")
plt.show()

# Swarm plot: Task-wise Final Score Distribution for Each Method with Colors
plt.figure(figsize=(12, 6))
sns.swarmplot(data=df, x="task_name", y="final_score", hue="method", dodge=True, palette=colors)
plt.xlabel("Task Name")
plt.ylabel("Final Score")
plt.title("Final Score Distribution by Task and Method")
plt.xticks(rotation=45)
plt.legend(title="Method")
plt.show()
