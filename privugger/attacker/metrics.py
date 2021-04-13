import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
import numpy as np
import pymc3 as pm
import scipy.stats as st
import pickle 
import datetime

class SimulationMetrics:
    """
    A method used to convert traces to something a data analyst can use to analyse
    """
    def __init__(self, traces=[], path=""):
        """
        Constructor for SimulationMetrics
        
        Parameters:
        ----------
        traces: List[Pymc3.trace]
            - A list of traces returned from simulation
        path: String
            - The path to a stored SimulationMetrics
        """
        if isinstance(traces, str):
            path = traces
            traces = []
        if len(traces):
            self.traces = traces
        else:
            self.traces = self.load_from_file(path)
        self.I = []
    
    def __str__(self):
        """
        Overrides the __str__ str method to print traces
        """
        return self.traces.__str__()

    def load_from_file(self, location):
        """
        A method which loads a pickled object as the trace

        Returns:
        ----------------
            - A "un"pickled object

        Parameters:
        ----------------
        Location: str
            - The location of the file
        """
        with open(location, "rb") as file:
            return pickle.load(file)

    def plot_mutual_bar(self, shift=0):
        """
        A method used in case of multiple parameters
        Is needed since the traces are than stored in a different sense

        Parameters
        ------------------
        Shift: int 
            - Since there are quite a large number of simulation a shift in the data can be needed

        """
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
        plt.tight_layout()
        plt.show()

    def plot_mutual_information(self, figsize=(16,8), as_bar=True):
        """
        Plots the mutual information as a graph for each distribution

        Parameters
        --------------
        figsize: Tuple<Int>
            - The size of the images
        as_bar: bool
            - Determines if the distribution should be a dot plot or a bar plot
        """
        I = self.mutual_information()
        size = len(I) if len(I) > 1 else 2
        plt.style.use('seaborn-darkgrid')
        _, ax = plt.subplots(2, 2,figsize=figsize)
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
                axs.annotate(str(best_info), xy=(x, best_info), fontsize=16)
                if str(k) == "TruncatedNormal":
                    labels.append("Truncated \n Normal")
                else:
                    labels.append(str(k))
                x+=1
            if as_bar:
                axs.bar([i for i in range(len(best_vals))], best_vals)
            alice = {0: (0,100), 1: (0,300), 2: (0,10), 3: (0,1)}
            pi_pos = "A_{\pi_" + str(pos+1) + "}"
            title = f"$I(Y_{pos+1};{pi_pos})$ for parameter {pos+1} where ${pi_pos}$ ~ $U$ {alice[pos]}"
            ylabel = f"$I(Y_{pos+1};{pi_pos})$"
            axs.set_title(title, fontsize=16)
            axs.set_ylabel(ylabel, fontsize=14)
            axs.set_xlabel(f"Distributions", fontsize=14)
            axs.set_xticks(range(len(labels)))
            axs.set_xticklabels(labels, fontsize=14)
            axs.set_ylim(0,ylim)
            pos += 1
        plt.tight_layout()
        plt.show()

    def highest_leakage(self, head=1, verbose=1):
        """
        A method used to calculate the highest leakage grouped by each distribution

        Returns
        -----------
        List[Tuple[Float, Tuple[String, ]]]
            - Returns a list containing the mutual information next to the specific distribution

        Parameters
        -----------
        head: Int
            - Determines how many distributions are included
        verbose: int
            - Detmines if the distributions should be printed
        """
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

    def save_to_file(self, location=""):
        """
        Save the particular trace to a file with the format:  Metrics-%Y-%m-%d-%H-%M-%S.priv

        Parameter:
        -----------
        location: string
            - The location in which the files should be saved
        """
        date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        with open(location+f"metrics-{date}.priv", "wb") as file:
            pickle.dump(self.traces, file)

    def mutual_information(self):
        """
        Calculates the mutual information for each distribution and appends them to a global variable I

        Returns
        --------
        List[float, List[String,]]
            - A list of the mutual information paired with its respective distribution
        """
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