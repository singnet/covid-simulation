# Simulation results

## Basic simulation scenarios

Simulation of scenarios described in the [article](https://towardsdatascience.com/agent-based-simulation-of-covid-19-health-and-economical-effects-6aa4ae0ff397)

### Scenario 1 - do nothing

![scenario1](scenario1.png 'scenario1')
[CSV data](scenario1.csv)

### Scenario 2 - restrict mobility only for symptomatic people

In this scenario we consider that 90% of symptomatic people are isolated.

```
symptomatic_isolation_rate = 0.9
```

![scenario2](scenario2.png 'scenario2')
[CSV data](scenario2.csv)

### Scenario 2.1 - restrict mobility only for symptomatic people (30% people uses a wearable which detects symptoms 1 day earlier)

```
symptomatic_isolation_rate = 0.9
weareable_adoption_rate = 0.3
```

![scenario2_1](scenario2_1.png 'scenario2_1')
[CSV data](scenario2_1.csv)

### Scenario 3 - restrict mobility for everyone

In this scenario we consider that 90% of symptomatic people and 80% of asymptomatic people are isolated.

```
symptomatic_isolation_rate = 0.9
asymptomatic_isolation_rate = 0.8
```

![scenario3](scenario3.png 'scenario3')
[CSV data](scenario3.csv)

### Scenario 4 - restrict mobility after 10% of the population being infected and release the restrictions when more then 95% is safe

In this scenario we use the same isolation rates used in scenario 3.

![scenario4](scenario4.png 'scenario4')
[CSV data](scenario4.csv)

## Simulations considering the use of masks

We've studied the effect of the use of masks in the rate of deaths during the
evolution of COVID19 infection in the population. The baseline for these
results is Scenario1 as described above.

We consider mask efficacy of X% to prevent that an infected person contamines a
non-infected one. We've simulated X in {30%, 40%, 50%, 60%, 70%}.

The mask also brings a negative effect by increasing the mobility of some of
the people using it, which tend to increase the infection rate and consequently
the rate of deaths.

The rate of mask-users that increase their mobility is ICR (Isolation Cheating
Rate) and the increase in their mobility is ICS (Isolation Cheating Severity).
I've simulated scenarios varying ICR and ICS in {10%, 20%, 30%, ..., 90%}.

In the charts, we plot the delta in percent points (pp) of the % of deaths
during the whole simulation compared to the baseline. This delta is coded in
color and color intensity. Negative deltas (less deaths) are plotted in blue
while positive deltas (more deaths) are plotted in yellow.

Mas adoption rate is the same in all scenarios (40%). Each scenario below
consider a different efficacy for the use of masks to prevent that an infected
person contamines a non-infected one.

### Mask efficacy: 30%

![maskefficacy30](mask_efficacy_30.png 'Mask efficacy: 30%')

### Mask efficacy: 40%

![maskefficacy40](mask_efficacy_40.png 'Mask efficacy: 40%')

### Mask efficacy: 50%

![maskefficacy50](mask_efficacy_50.png 'Mask efficacy: 50%')

### Mask efficacy: 60%

![maskefficacy60](mask_efficacy_60.png 'Mask efficacy: 60%')


## Simulations considering the use of masks and a wearable which allows earlier detection of symptoms

In this simulation we consider that 30% of people use a wearable with an app which detects COVID-19 symptoms 1 day earlier.

We consider the same scenarios and parameters of the previous simulation.

### Mask efficacy: 30%

![maskefficacy30](wearable_mask_efficacy_30.png 'Mask efficacy: 30%')

### Mask efficacy: 40%

![maskefficacy40](wearable_mask_efficacy_40.png 'Mask efficacy: 40%')

### Mask efficacy: 50%

![maskefficacy50](wearable_mask_efficacy_50.png 'Mask efficacy: 50%')

### Mask efficacy: 60%

![maskefficacy60](wearable_mask_efficacy_60.png 'Mask efficacy: 60%')


