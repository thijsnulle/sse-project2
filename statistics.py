import csv, math, re
from dataclasses import dataclass
from scipy.stats import ttest_ind
import numpy as np

@dataclass
class CarbonEmission:
    model_id: str
    datasets_size: int
    co2_emission: float
    co2_reported: float
    geographical_location: str
    accuracy: float
    f1: float
    rouge_1: float
    rouge_l: float
    domain: str
    size: int
    auto: bool

def strip():
    with open('HFCO2.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)

        carbon_emissions = []
        for row in reader:
            performance_regex = re.search(r"\{'accuracy': (.*), 'f1': (.*), 'rouge1': (.*), 'rougeL': (.*)\}", row[9])
            performance_metrics = {
                'accuracy': float(performance_regex.group(1)),
                'f1': float(performance_regex.group(2)),
                'rouge1': float(performance_regex.group(3)),
                'rougeL': float(performance_regex.group(4)),
            }

            carbon_emissions.append(CarbonEmission(
                model_id=row[0],
                datasets_size=int(float(row[2] or 0)),
                co2_emission=float(row[3]),
                co2_reported=float(row[4]),
                geographical_location=row[7],
                accuracy=performance_metrics['accuracy'],
                f1=performance_metrics['f1'],
                rouge_1=performance_metrics['rouge1'],
                rouge_l=performance_metrics['rougeL'],
                domain=row[14],
                size=int(float(row[15] or 0)),
                auto=row[19] == 'True',
            ))

    with open('co2_data.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(CarbonEmission.__dataclass_fields__.keys())

        for carbon_emission in carbon_emissions:
            writer.writerow(carbon_emission.__dict__.values())

def load() -> list[CarbonEmission]:
    carbon_emissions = []
    with open('co2_data.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            carbon_emissions.append(CarbonEmission(
                model_id=row[0],
                datasets_size=int(row[1]),
                co2_emission=float(row[2]),
                co2_reported=float(row[3]),
                geographical_location=row[4],
                accuracy=float(row[5]),
                f1=float(row[6]),
                rouge_1=float(row[7]),
                rouge_l=float(row[8]),
                domain=row[9],
                size=int(row[10]),
                auto=row[11] == 'True',
            ))

    return list({x.model_id.strip().lower(): x for x in carbon_emissions}.values())

def format_large_number(number):
    suffixes = ['k', 'M', 'B']
    
    if number < 1000:
        return str(number)
    
    for suffix in suffixes:
        number /= 1000.0
        if abs(number) < 1000:
            return f"{number:.1f}{suffix}"

    return str(number)

def calculate_probability(N, M, k, top_models):
    """
    Calculate the probability of observing 'top_models' or more non-auto models among the top k emission models.
    
    Parameters:
        N (int): Total number of models.
        M (int): Number of non-auto models.
        k (int): Number of top emission models considered.
        top_models (int): Number of top models to consider.
    
    Returns:
        float: Probability of observing 'top_models' or more non-auto models among the top k emission models.
    """
    probability = 0
    for x in range(top_models, min(M, k) + 1):  # Calculate from 'top_models' to min(M, k)
        numerator = math.comb(M, x) * math.comb(N - M, k - x)
        denominator = math.comb(N, k)
        probability += numerator / denominator
    return probability

def comparison():
    carbon_emissions = load()
    carbon_emissions = [x for x in carbon_emissions if x.size > 0 and x.co2_emission > 0]

    sorted_emissions = sorted(carbon_emissions, key=lambda x: x.co2_emission / x.size)

    first_auto_index = next((i for i, x in enumerate(sorted_emissions) if x.auto), None)
    first_non_auto_index = next((i for i, x in enumerate(sorted_emissions) if not x.auto), None)

    last_auto_index = next((i for i, x in enumerate(reversed(sorted_emissions)) if x.auto), None)
    last_non_auto_index = next((i for i, x in enumerate(reversed(sorted_emissions)) if not x.auto), None)

    print('===== Auto vs. Non-Auto =====')
    if first_auto_index > first_non_auto_index:
        print("Position of most efficient auto model:", first_auto_index + 1)
        print(f'Probability of observing 1 or more non-auto models among the top {first_auto_index} emission models:', calculate_probability(len(carbon_emissions), sum(1 for x in carbon_emissions if not x.auto), first_auto_index, 1))
    else:
        print("Position of most efficient non-auto model:", first_non_auto_index + 1)
        print(f'Probability of observing 1 or more non-auto models among the top {first_non_auto_index} emission models:', calculate_probability(len(carbon_emissions), sum(1 for x in carbon_emissions if not x.auto), first_non_auto_index, 1))
    print()


    if last_auto_index > last_non_auto_index:
        print("Position of least efficient auto model:", last_auto_index + 1)
        print(f'Probability of observing 1 or more non-auto models among the top {last_auto_index} emission models:', calculate_probability(len(carbon_emissions), sum(1 for x in carbon_emissions if not x.auto), last_auto_index, 1))
    else:
        print("Position of least efficient non-auto model:", last_non_auto_index + 1)
        print(f'Probability of observing 1 or more non-auto models among the top {last_non_auto_index} emission models:', calculate_probability(len(carbon_emissions), sum(1 for x in carbon_emissions if not x.auto), last_non_auto_index, 1))

    print()
    print('===== Top 25 most efficient models =====')
    print('Rank\tModel ID\t\tCO2 Emission\tDataset Size\tAuto')
    for i, carbon_emission in enumerate(sorted_emissions[:25]):
        value_str = '{:.16e}'.format(carbon_emission.co2_emission / carbon_emission.size)
        mantissa, exponent = value_str.split('e')

        dataset_size = format_large_number(carbon_emission.size)

        print(f'{str(i+1).rjust(2)}.\t{carbon_emission.model_id[:20].ljust(20)}\t{float(mantissa):.2f}e{exponent}\t{dataset_size}\t\t{'✓' if carbon_emission.auto else ''}')

    print()
    print('===== Top 25 least efficient models =====')
    print('Rank\tModel ID\t\tCO2 Emission\tDataset Size\tAuto')
    for i, carbon_emission in enumerate(reversed(sorted_emissions[-25:])):
        value_str = '{:.16e}'.format(carbon_emission.co2_emission / carbon_emission.size)
        mantissa, exponent = value_str.split('e')

        dataset_size = format_large_number(carbon_emission.size)

        print(f'{str(i+1).rjust(2)}.\t{carbon_emission.model_id[:20].ljust(20)}\t{float(mantissa):.2f}e{exponent}\t{dataset_size}\t\t{'✓' if carbon_emission.auto else ''}')

    total_non_auto = sum(1 for x in carbon_emissions if not x.auto)
    total_auto = sum(1 for x in carbon_emissions if x.auto)

    print()
    print('===== Non-Auto vs Auto =====')
    print("Total Non-Automatic Models:", total_non_auto)
    print("Total Automatic Models:", total_auto)

    auto_emissions = [np.log(x.co2_emission / x.size) for x in carbon_emissions if x.auto]
    non_auto_emissions = [np.log(x.co2_emission / x.size) for x in carbon_emissions if not x.auto]
    t, p = ttest_ind(auto_emissions, non_auto_emissions, equal_var=True)

    print()
    print('===== T-Test =====')
    print("T-Value:", t)
    print("P-Value:", p)

    print()
    if p < 0.05:
        print("There is a statistically significant difference between automatic and non-automatic model emissions.")
    else:
        print("There is no statistically significant difference between automatic and non-automatic model emissions.")

if __name__ == '__main__':
    comparison()
