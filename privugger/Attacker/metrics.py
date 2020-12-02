import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
import numpy as np
import pymc3 as pm
import scipy.stats as st
import pickle 
import datetime

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

    def plot_mutual_bar(self, shift=0):
        start, stop = 0,10
        executions = 5 #len(sm.traces)//2//10
        fig, ax = plt.subplots(executions,2, figsize=(20,18))
        for j in range(shift, shift+executions):
            for p in range(2):
                start = (p*10)+(j*20)
                stop = ((p*10)+10)+(j*20)
                traces = self.traces[start:stop]
                Is = {}
                labels = []
                for i in range(10):
                    trace = traces[i][0]
                    name = traces[i][2][0]
                    alice = trace[f"intDist_{i}"]
                    output = trace[f"Output_{i}"]
                    I = mutual_info_regression([[a] for a in alice], output, discrete_features=True)[0]
                    if name[0] in Is:
                        Is[name[0]].append(I)
                    else:
                        labels.append(name[0])
                        Is[name[0]] = [I]
                values = len(Is.keys())
                vals = [max(v[1]) for v in Is.items()]
                ax[j-shift][p].bar(labels, vals)
                ax[j-shift][p].set_ylim(0,7)
                ax[j-shift][p].set_title(f"Parameter {p} opposite was {self.traces[start+stop//2][2][1]}")
                ax[j-shift][p].set_ylabel("$I(X;Y)$")
            print("\r" + str(j) , end="\r")
        plt.tight_layout()
        plt.show()

    def plot_mutual_information(self, figsize=(16,8), as_bar=True):
        """
        Plots the mutual information as a graph for each distribution
        """
        I = self.mutual_information()
        size = len(I) if len(I) > 1 else 2
        _, ax = plt.subplots(size, 1,figsize=figsize)
        ylim = round(max(self.highest_leakage(head=1, verbose=0), key=lambda x: x[0])[0][0]+0.5)
        for pos, (axs, values) in enumerate(zip(ax.flatten(), I)):
            x = 0
            items = values.items()
            labels = []
            best_vals = []
            for k,v in items:
                if as_bar:
                    best = max(v, key=lambda x: x[0])
                    best_vals.append(best[0])
                else:
                    for value, info in v:
                        axs.plot([x], value, "x", label=str(info[1:]))
                best_info = round(max(v, key=lambda x: x[0])[0],2)
                axs.annotate(str(best_info), xy=(x, best_info))
                labels.append(str(k))
                x+=1
            if as_bar:
                axs.bar([i for i in range(len(best_vals))], best_vals)
            alice = (0,51) if pos == 0 else (0,100)
            axs.set_title(f"Mutual information for parameter {pos} where A ~ $U$ {alice}")
            axs.set_ylabel("Mutual Information $I(A:Y)$ \n higher values indicate higher leakage")
            axs.set_xlabel(f"Distributions")
            axs.set_xticks(range(len(labels)))
            axs.set_xticklabels(labels, fontsize=12)
            axs.set_ylim(0,ylim)
            pos += 1
        plt.tight_layout()
        plt.show()

    def highest_leakage(self, head=1, verbose=1):
        if not len(self.I):
            self.mutual_information()
        best_vals = []
        for parameter_pos, l in enumerate(self.I):
            best_dist = []
            for k,v in l.items():
                best = max(v, key=lambda x: x[0])
                best_dist.append(best)
            best_dist = list(sorted(best_dist, key=lambda x: x[0], reverse=True))
            if verbose:
                print(f"The distribution that had the most leakage was {best_dist[:head]} for parameter {parameter_pos}")
            best_vals.append(best_dist[:head])
        return best_vals

    def plot_distributions(self):
        best = self.highest_leakage()
        _, ax = plt.subplots(len(best),1)
        for parameter_pos, best_vals in enumerate(best):
            l,h = (0,100) if parameter_pos else (0,51)
            x = np.linspace(l,h,(h-l)*20)
            alice_pmf = [1.0 / (h - l + 1)] * len(x)
            ax[parameter_pos].plot(x, alice_pmf, label=f"Alice ~ $U$(0,51)")
            for dist in best_vals:
                pmf, name = self.parse_dist(x, dist)
                ax[parameter_pos].plot(x, pmf, label=name)
            ax[parameter_pos].legend()
        plt.show()

    def parse_dist(self, x, dist):
        name = dist[1][0]
        parameters = dist[1][1:]
        if name == "Poisson":
            return (st.poisson.pmf(x, parameters[0]), f"Poisson: $\mu$ = {parameters[0]}")
        elif name == "Beta":
            return (st.beta.pdf(x, parameters[0], parameters[1]), r"Beta: $\alpha$ = {}, $\beta$ = {}".format(parameters[0], parameters[1]))
        else:
            return (x, "")

    def save_to_file(self, location):
        date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        with open(location+f"metrics-{date}.priv", "wb") as file:
            pickle.dump(self.traces, file)

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