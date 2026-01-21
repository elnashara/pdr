import csv
import math
import statistics
from typing import List, Dict, Any

# Optional: for inferential stats like t-tests
try:
    from scipy.stats import ttest_rel, wilcoxon
except ImportError:
    # If scipy isn't installed, handle gracefully or request the user to install it
    print("Warning: scipy not installed. Inferential tests won't be available.")

class ExperimentAnalyzer:
    """
    Loads CSV results and performs descriptive/inferential statistics on them.
    """

    def load_results(self, filename: str) -> List[Dict[str, Any]]:
        """
        Loads results from the CSV file into a list of dictionaries, where each
        row is one dictionary. Returns the list.
        """
        data = []
        with open(filename, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields from strings to floats/ints as needed
                row["iteration_count"] = int(row["iteration_count"])
                row["time_spent_sec"] = float(row["time_spent_sec"])
                row["final_score"] = float(row["final_score"])
                # If expert or subjective fields exist, parse them similarly
                # e.g. row["expert_correctness_score"] = float(...) if not empty
                data.append(row)
        return data

    def descriptive_stats(self, data: List[Dict[str, Any]], measure: str) -> Dict[str, float]:
        """
        Computes descriptive statistics (mean, stdev) for the given measure (e.g., 'iteration_count')
        in the list of result dictionaries. Returns a dict with keys like 'mean' and 'stdev'.
        """
        values = [row[measure] for row in data if not math.isnan(row[measure])]
        if len(values) == 0:
            return {"mean": None, "stdev": None, "count": 0}

        mean_val = statistics.mean(values)
        stdev_val = statistics.pstdev(values) if len(values) > 1 else 0.0
        return {
            "mean": mean_val,
            "stdev": stdev_val,
            "count": len(values)
        }

    def compare_two_conditions(
        self,
        data1: List[Dict[str, Any]],
        data2: List[Dict[str, Any]],
        measure: str,
        test_type: str = "paired_t"
    ) -> Dict[str, Any]:
        """
        Performs an inferential test comparing two conditions (e.g., baseline vs. PDR)
        on the given measure (e.g., 'iteration_count'). The data1 and data2 lists should be
        aligned or carefully matched (e.g., same participants/tasks in the same order).
        
        test_type can be 'paired_t' or 'wilcoxon'.

        Returns a dict with the test statistic and p-value.
        """
        # Extract the measure for each row
        vals1 = [row[measure] for row in data1]
        vals2 = [row[measure] for row in data2]

        # Quick check for length mismatch
        if len(vals1) != len(vals2):
            return {"error": "Data1 and Data2 length mismatch"}

        result = {}

        if test_type == "paired_t":
            # Paired t-test
            stat, p_value = ttest_rel(vals1, vals2)
            result["test"] = "paired t-test"
            result["statistic"] = stat
            result["p_value"] = p_value
        elif test_type == "wilcoxon":
            # Wilcoxon signed-rank test
            stat, p_value = wilcoxon(vals1, vals2)
            result["test"] = "Wilcoxon signed-rank test"
            result["statistic"] = stat
            result["p_value"] = p_value
        else:
            result["error"] = f"Unsupported test_type: {test_type}"

        return result

    def group_by_key(self, data: List[Dict[str, Any]], group_key: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Groups the data by the specified key (e.g. 'participant_name') and returns a dict
        mapping key_value -> list of row dicts.
        """
        grouped = {}
        for row in data:
            val = row[group_key]
            if val not in grouped:
                grouped[val] = []
            grouped[val].append(row)
        return grouped
