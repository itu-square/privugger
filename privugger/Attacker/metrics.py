import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
import numpy as np
import pymc3 as pm
import pickle 

class SimulationMetrics:
    def __init__(self, traces=[], path=""):
        if len(traces):
            self.traces = traces
        else:
            self.traces = self.load_from_file(path)
        self.I = []
    
    def __str__(self):
        return self.traces.__str__()

    def load_from_file(self, location):
        with open(location, "rb") as file:
            return pickle.load(file)

    def plot_mutual_information(self, figsize=(16,8)):
        """
        Plots the mutual information as a graph for each distribution
        """
        I = self.mutual_information()
        size = len(I) if len(I) > 1 else 2
        _, ax = plt.subplots(size, 1,figsize=figsize)
        for pos, (axs, values) in enumerate(zip(ax.flatten(), I)):
            x = 0
            items = values.items()
            labels = []
            for k,v in items:
                for value, info in v:
                    axs.plot([x], value, "x", label=str(info[1:]))
                best_info = round(max(v, key=lambda x: x[0])[0],2)
                axs.annotate(str(best_info), xy=(x, best_info))
                labels.append(str(k))
                x += 1
            axs.set_title(f"Mutual information for parameter {pos}")
            axs.set_ylabel(f"Mutual Information")
            axs.set_xlabel(f"Distributions")
            axs.set_xticks(range(len(labels)))
            axs.set_xticklabels(labels, fontsize=12)
            pos += 1
        plt.tight_layout()
        plt.show()

    def highest_leakage(self):
        ...

    def save_to_file(self, location):
        with open(location+"metrics2.priv", "wb") as file:
            pickle.dump(self.traces, file)
    
    def plot_distributions(self):
        size = len(self.traces[0][1])+1
        fig,ax = plt.subplots(size)
        for trace, info, _ in self.traces:
            info.append("Output")
            for axs, name in zip(ax, info):
                pm.plot_posterior(trace[name], ax=axs)
        plt.show()
    
    def mutual_information(self):
        _, names, _ = self.traces[0]
        size = len(names)
        mutual_information = [{} for _ in range(size)]
        for i in range(size):
            for trace, names, info in self.traces:
                discrete = ("int" in str(names[i]))
                alice = trace[names[i]]
                try:
                    output = trace["Output"]
                except:
                    print(names[i])
                    pos = int(str(names[i]).split("_")[-1])
                    if pos < 10:
                        output = trace[f"Output_{pos}"]
                    else:
                        continue
                I_ao = mutual_info_regression([[j] for j in alice], output, discrete_features=discrete)[0]
                while (len(info) == 1):
                    info = info[0] # Used to unwrap the inner information in case of subtypes such as List[List[Tuple[...]]]
                if isinstance(info, tuple) or (isinstance(info, list) and isinstance(info[0], list)):
                    info = info[i]
                if info[0] in mutual_information[i]:
                    mutual_information[i][info[0]].append((I_ao,info))
                else:
                    mutual_information[i][info[0]] = [(I_ao, info)]
        self.I = mutual_information
        return mutual_information