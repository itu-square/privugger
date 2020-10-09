import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
import numpy as np

class SimulationMetrics:
    def __init__(self, traces):
        self.traces = traces
    
    def __str__(self):
        return self.traces.__str__()

    def plot_mutual_information(self):
        """
        Plots the mutual information as a graph for each distribution
        """
        I = self.mutual_information()
        print(I)
        size = len(I) if len(I) > 1 else 2
        _, ax = plt.subplots(size, 1)
        for pos, (axs, values) in enumerate(zip(ax.flatten(), I)):
            x = 0
            items = values.items()
            labels = []
            for k,v in items:
                for value, info in v:
                    axs.plot([x], value, "x", label=str(info[1:]))
                labels.append(str(k))
                x += 1
            axs.set_title(f"Mutual information for parameter {pos}")
            axs.set_ylabel(f"Mutual Information")
            axs.set_ylim(0,1)
            axs.set_xlabel(f"Distributions")
            axs.set_xticks(range(len(labels)))
            axs.set_xticklabels(labels, fontsize=12)
            pos += 1
        plt.tight_layout()
        plt.show()

            
            
    def save_to_file(self):
        ...
    
    def plot_distributions(self):
        ...
    
    def mutual_information(self):
        _, names, _ = self.traces[0]
        size = len(names)
        mutual_information = [{} for _ in range(size)]
        for i in range(size):
            for trace, names, info in self.traces:
                discrete = ("int" in str(names[i]))
                alice = trace[names[i]]
                output = trace["Output"]
                I_ao = mutual_info_regression([[j] for j in alice], output, discrete_features=discrete)[0]
                I_aa = mutual_info_regression([[j] for j in alice], alice, discrete_features=discrete)[0]
                while (len(info) == 1):
                    info = info[0] # Used to unwrap the inner information in case of subtypes such as List[List[Tuple[...]]]
                if isinstance(info, tuple):
                    info = info[i]
                if info[0] in mutual_information[i]:
                    mutual_information[i][info[0]].append((I_ao/I_aa, info))
                else:
                    mutual_information[i][info[0]] = [(I_ao/I_aa, info)]
        return mutual_information