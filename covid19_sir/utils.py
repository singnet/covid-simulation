import matplotlib.pyplot as plt
import pandas as pd
from model.base import CovidModel, get_parameters
from model.location import Location

class BasicStatistics():
    def __init__(self, model):
        self.susceptible = []
        self.infected = []
        self.recovered = []
        self.hospitalization = []
        self.icu = []
        self.death = []
        self.income = [1.0]
        self.cycles_count = 0
        self.covid_model = model

    def start_cycle(self, model):
        self.cycles_count += 1
        pop = self.covid_model.global_count.total_population
        work_pop = self.covid_model.global_count.work_population
        #print(f"infected = {self.covid_model.global_count.infected_count} recovered = {self.covid_model.global_count.recovered_count}")
        self.susceptible.append(self.covid_model.global_count.susceptible_count / pop)
        self.infected.append(self.covid_model.global_count.infected_count / pop)
        self.recovered.append(self.covid_model.global_count.recovered_count / pop)
        self.hospitalization.append((self.covid_model.global_count.total_hospitalized) / pop)
        self.icu.append(self.covid_model.global_count.high_severity_count / pop)
        self.death.append(self.covid_model.global_count.death_count / pop)
        self.income.append(self.covid_model.global_count.total_income / work_pop)

    def end_cycle(self, model):
        pass

    def export_chart(self, fname):
        self.income.pop(1)
        df = pd.DataFrame(data={
            'Susceptible': self.susceptible,
            'Infected': self.infected,
            'Recovered': self.recovered,
            'Death': self.death,
            'Hospitalization': self.hospitalization,
            'Severe': self.icu,
            'Income': self.income
        })
        color = {
            'Susceptible' : 'lightblue',
            'Infected': 'gray',
            'Recovered': 'lightgreen',
            'Death': 'black',
            'Hospitalization': 'orange',
            'Severe': 'red',
            'Income': 'magenta'
        }
        fig, ax = plt.subplots()
        ax.set_title('Contagion Evolution')
        ax.set_xlim((0, self.cycles_count))
        ax.axhline(y=get_parameters().get('hospitalization_capacity'), c="black", ls='--', label='Critical limit')
        for col in df.columns.values:
            ax.plot(df.index.values, df[col].values, c=color[col], label=col)
        ax.set_xlabel("Days")
        ax.set_ylabel("% of Population")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper right')
        fig.savefig(fname)

    def export_csv(self, fname):
        df = pd.DataFrame(data={
            'Susceptible': self.susceptible,
            'Infected': self.infected,
            'Recovered': self.recovered,
            'Death': self.death,
            'Hospitalization': self.hospitalization,
            'Severe': self.icu,
            'Income': self.income
        })
        df.to_csv(fname)
