import pandas as pd
import json
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive # Import drive at the top level
import time
import os
import traceback

def OptUsingGA(json_file_path, jsonfilename, display_graphs=1):
    try:
        # --- 1. Data Loading and Initial Parameter Extraction ---
        try:
            with open(json_file_path, 'r') as f:
                json_data_dict = json.load(f)
            print(f'Successfully loaded "{json_file_path}" into a Python dictionary.')
        except FileNotFoundError:
            print(f'Error: The file "{json_file_path}" was not found.')
            return None, None
        except json.JSONDecodeError as e:
            print(f'Error: Could not decode JSON from "{json_file_path}". Error: {e}')
            return None, None
        except Exception as e:
            print(f'An unexpected error occurred during JSON loading: {e}')
            return None, None

        # Create DataFrame for storage details
        df_storage_details = pd.DataFrame({
            "Storage_levels": json_data_dict["Storage_levels"],
            "Minimum_units_per_storage_level": json_data_dict["Minimum_units_per_storage_level"],
            "Maximum_units_per_storage_level": json_data_dict["Maximum_units_per_storage_level"],
            "Cost_per_storage_level": json_data_dict["Cost_per_storage_level"]
        })

        # Create DataFrame for periods, demand, and production rates
        df_periods_demand_products = pd.DataFrame({
            "Period": json_data_dict["Periods"],
            "Demand_per_period": json_data_dict["Demand_per_period"],
            "Products_units_per_machine_per_period": json_data_dict["Products_units_per_machine_per_period"]
        })

        # Create dictionary for financial and machine details
        financial_and_machine_details_dict = {
            'Initial_inventory': json_data_dict['Initial_inventory'],
            'Initial_machine_number': json_data_dict['Initial_machine_number'],
            'Maximum_number_of_machines': json_data_dict['Maximum_number_of_machines'],
            'Cost_to_buy_machine': json_data_dict['Cost_to_buy_machine'],
            'Cost_to_remove_machine': json_data_dict['Cost_to_remove_machine'],
            'Cost_to_produce_one_unit_of_product': json_data_dict['Cost_to_produce_one_unit_of_product']
        }

        # Extract parameters to local variables for the function
        T = json_data_dict['Periods']
        S = json_data_dict['Storage_levels']

        demand = df_periods_demand_products.set_index('Period')['Demand_per_period'].to_dict()
        production_rate = df_periods_demand_products.set_index('Period')['Products_units_per_machine_per_period'].to_dict()

        min_storage_units = df_storage_details.set_index('Storage_levels')['Minimum_units_per_storage_level'].to_dict()
        max_storage_units = df_storage_details.set_index('Storage_levels')['Maximum_units_per_storage_level'].to_dict()

        initial_inventory = financial_and_machine_details_dict['Initial_inventory']
        initial_machines = financial_and_machine_details_dict['Initial_machine_number']
        max_machines = financial_and_machine_details_dict['Maximum_number_of_machines']
        cost_buy_machine = financial_and_machine_details_dict['Cost_to_buy_machine']
        cost_remove_machine = financial_and_machine_details_dict['Cost_to_remove_machine']
        cost_produce_unit = financial_and_machine_details_dict['Cost_to_produce_one_unit_of_product']

        cost_store_unit_per_period = json_data_dict['Cost_to_store_one_unit_of_product_per_period']
        cost_storage_level = dict(zip(json_data_dict['Storage_levels'], json_data_dict['Cost_per_storage_level']))

        print("Extracted parameters for Genetic Algorithm.")

        # --- 2. Prepare Genetic Algorithm Framework ---

        def setup_ga_parameters():
            """Defines and returns common genetic algorithm parameters."""
            ga_params = {
                "POPULATION_SIZE": 100,
                "NUM_GENERATIONS": 50,
                "MUTATION_RATE": 0.1,
                "CROSSOVER_RATE": 0.7
            }
            print("\nGenetic algorithm parameters defined:")
            for param, value in ga_params.items():
                print(f"  {param}: {value}")
            return ga_params

        ga_parameters = setup_ga_parameters()

        # --- 3. Implement Fitness Function ---

        def evaluate_chromosome(individual):
            """Evaluates a chromosome (active machine numbers) by simulating the planning horizon,
            calculating total costs and max storage level, and incorporating penalties.

            Args:
                individual (list): A list representing a chromosome, where each element is the
                                   number of active machines for a period.

            Returns:
                tuple: A tuple containing the weighted fitness value (total_objective,).
            """
            total_costs = 0.0
            total_penalties = 0.0
            current_inventory = float(initial_inventory)
            current_machines = float(initial_machines)
            max_inventory_observed = 0.0

            INFEASIBILITY_PENALTY = 1e6  # Large penalty for constraint violations
            WEIGHT_MAX_STORAGE = 1000.0  # Weight for max_inventory_observed in fitness

            if len(individual) != len(T):
                return INFEASIBILITY_PENALTY * (len(T) + 1), # High penalty for incorrect length

            for t_idx, t in enumerate(T):
                M_t_chromosome_raw = individual[t_idx]

                if M_t_chromosome_raw < 0 or M_t_chromosome_raw > max_machines:
                    total_penalties += INFEASIBILITY_PENALTY
                    M_t_chromosome = max(0, min(max_machines, int(round(M_t_chromosome_raw))))
                else:
                    M_t_chromosome = int(round(M_t_chromosome_raw))

                machines_bought = max(0.0, M_t_chromosome - current_machines)
                machines_removed = max(0.0, current_machines - M_t_chromosome)

                total_costs += cost_buy_machine * machines_bought
                total_costs += cost_remove_machine * machines_removed

                current_machines = M_t_chromosome

                production_t = current_machines * production_rate[t]
                total_costs += cost_produce_unit * production_t

                ending_inventory_t = current_inventory + production_t - demand[t]

                if ending_inventory_t < 0:
                    total_penalties += abs(ending_inventory_t) * INFEASIBILITY_PENALTY
                    ending_inventory_t = 0.0

                current_inventory = ending_inventory_t
                total_costs += cost_store_unit_per_period * current_inventory

                chosen_s = None
                min_cost_for_storage = float('inf')
                for s_level in S:
                    if min_storage_units[s_level] <= current_inventory <= max_storage_units[s_level]:
                        if cost_storage_level[s_level] < min_cost_for_storage:
                            min_cost_for_storage = cost_storage_level[s_level]
                            chosen_s = s_level

                if chosen_s is None:
                    total_penalties += INFEASIBILITY_PENALTY
                    chosen_s = S[0] # Default for display if no fit
                total_costs += cost_storage_level[chosen_s]

                max_inventory_observed = max(max_inventory_observed, current_inventory)

            total_objective = total_costs + total_penalties + (WEIGHT_MAX_STORAGE * max_inventory_observed)

            return total_objective,

        print("Fitness function `evaluate_chromosome` defined.")

        # --- 4. Implement Genetic Operators (NumPy-based) ---

        def initial_chromosome_generation(population_size, chromosome_length):
            """Generates an initial population of chromosomes (active machine numbers)."""
            population = np.random.randint(0, max_machines + 1, size=(population_size, chromosome_length))
            return population

        def np_selection(population, fitness_values, k=3):
            """Performs tournament selection."""
            selected_parents = []
            for _ in range(population.shape[0]):
                tournament_indices = np.random.choice(population.shape[0], k, replace=False)
                tournament_fitness = fitness_values[tournament_indices]
                winner_idx_in_tournament = np.argmin(tournament_fitness)
                winner_original_idx = tournament_indices[winner_idx_in_tournament]
                selected_parents.append(population[winner_original_idx])
            return np.array(selected_parents)

        def np_crossover(parent1, parent2):
            """Performs two-point crossover between two parent chromosomes."""
            chromosome_length = len(parent1)
            if chromosome_length < 2:
                return parent1.copy(), parent2.copy()

            points = np.random.choice(chromosome_length - 1, 2, replace=False)
            point1, point2 = sorted(points + 1)

            offspring1 = np.concatenate((parent1[:point1], parent2[point1:point2], parent1[point2:]))
            offspring2 = np.concatenate((parent2[:point1], parent1[point1:point2], parent2[point2:]))

            return offspring1, offspring2

        def np_mutation(chromosome, mutation_rate, low_bound, upper_bound):
            """Performs uniform integer mutation on a chromosome."""
            mutated_chromosome = chromosome.copy()
            for i in range(len(mutated_chromosome)):
                if np.random.rand() < mutation_rate:
                    mutated_chromosome[i] = np.random.randint(low_bound, upper_bound + 1)
            return mutated_chromosome

        print("NumPy-based genetic operators (initial_chromosome_generation, np_selection, np_crossover, np_mutation) defined.")

        # --- 5. Run the Genetic Algorithm ---

        POPULATION_SIZE = ga_parameters["POPULATION_SIZE"]
        NUM_GENERATIONS = ga_parameters["NUM_GENERATIONS"]
        MUTATION_RATE = ga_parameters["MUTATION_RATE"]
        CROSSOVER_RATE = ga_parameters["CROSSOVER_RATE"]

        chromosome_length = len(T)

        print(f"\nExecuting Genetic Algorithm for {NUM_GENERATIONS} generations with population size {POPULATION_SIZE}")
        print(f"Mutation Rate: {MUTATION_RATE}, Crossover Rate: {CROSSOVER_RATE}")

        population = initial_chromosome_generation(POPULATION_SIZE, chromosome_length)

        best_individual = None
        best_fitness = float('inf')

        start_time = time.time()
        for gen in range(NUM_GENERATIONS):
            fitness_values = np.array([evaluate_chromosome(ind)[0] for ind in population])

            current_best_idx = np.argmin(fitness_values)
            current_best_fitness = fitness_values[current_best_idx]

            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[current_best_idx].copy()

            parents = np_selection(population, fitness_values)

            offspring = []
            for i in range(0, POPULATION_SIZE, 2):
                parent1 = parents[i]
                if i + 1 < POPULATION_SIZE:
                    parent2 = parents[i+1]
                    if random.random() < CROSSOVER_RATE:
                        child1, child2 = np_crossover(parent1, parent2)
                        offspring.append(child1)
                        offspring.append(child2)
                    else:
                        offspring.append(parent1.copy())
                        offspring.append(parent2.copy())
                else:
                    offspring.append(parent1.copy())

            offspring = offspring[:POPULATION_SIZE]

            mutated_offspring = np.array([
                np_mutation(child, MUTATION_RATE, 0, max_machines) for child in offspring
            ])

            population = mutated_offspring

            print(f"Generation {gen+1}/{NUM_GENERATIONS}: Best Fitness = {best_fitness:.2f}")
        end_time = time.time()
        ga_runtime = end_time - start_time

        print(f"\nGenetic Algorithm Finished.")
        print(f"Best Chromosome (Active Machines per Period): {best_individual}")
        print(f"Best Fitness (Weighted Cost): {best_fitness:.2f}")

        # --- 6. Analyze and Present Genetic Algorithm Results ---

        def get_ga_optimal_plan_details(individual):
            """Re-evaluates the best chromosome to get detailed optimal plan and costs."""
            production_values = {}
            inventory_values = {}
            machine_values = {}
            machine_bought_values = {}
            machine_removed_values = {}
            storage_choice_data = []

            current_inventory = float(initial_inventory)
            current_machines = float(initial_machines)
            max_inventory_observed = 0.0

            # Separate cost components for detailed tracking
            production_cost_comp = 0.0
            machine_buying_cost_comp = 0.0
            machine_removal_cost_comp = 0.0
            inventory_holding_cost_comp = 0.0
            storage_level_fixed_cost_comp = 0.0


            for t_idx, t in enumerate(T):
                M_t_chromosome_raw = individual[t_idx]

                M_t_chromosome = max(0, min(max_machines, int(round(M_t_chromosome_raw))))

                machines_bought = max(0.0, M_t_chromosome - current_machines)
                machines_removed = max(0.0, current_machines - M_t_chromosome)

                machine_buying_cost_comp += cost_buy_machine * machines_bought
                machine_removal_cost_comp += cost_remove_machine * machines_removed

                current_machines = M_t_chromosome

                production_t = current_machines * production_rate[t]
                production_cost_comp += cost_produce_unit * production_t

                ending_inventory_t = current_inventory + production_t - demand[t]

                if ending_inventory_t < 0:
                    ending_inventory_t = 0.0

                current_inventory = ending_inventory_t
                inventory_holding_cost_comp += cost_store_unit_per_period * current_inventory

                chosen_s = None
                min_cost_for_storage = float('inf')
                for s_level in S:
                    if min_storage_units[s_level] <= current_inventory <= max_storage_units[s_level]:
                        if cost_storage_level[s_level] < min_cost_for_storage:
                            min_cost_for_storage = cost_storage_level[s_level]
                            chosen_s = s_level

                if chosen_s is None:
                    chosen_s = S[0] # Default for display if no fit
                storage_level_fixed_cost_comp += cost_storage_level[chosen_s]

                max_inventory_observed = max(max_inventory_observed, current_inventory)

                production_values[t] = production_t
                inventory_values[t] = current_inventory
                machine_values[t] = current_machines
                machine_bought_values[t] = machines_bought
                machine_removed_values[t] = machines_removed
                storage_choice_data.append({
                    'Period': t,
                    'Storage_Level': chosen_s,
                    'Min_Units': min_storage_units[chosen_s],
                    'Max_Units': max_storage_units[chosen_s],
                    'Storage_Cost': cost_storage_level[chosen_s]
                })
            
            actual_total_cost = production_cost_comp + machine_buying_cost_comp + \
                                machine_removal_cost_comp + inventory_holding_cost_comp + \
                                storage_level_fixed_cost_comp

            return {
                'production': production_values,
                'inventory': inventory_values,
                'active_machines': machine_values,
                'machines_bought': machine_bought_values,
                'machines_removed': machine_removed_values,
                'storage_choices': storage_choice_data,
                'max_inventory_observed': max_inventory_observed,
                'total_costs_ga_actual': actual_total_cost,
                'total_production_cost': production_cost_comp,
                'total_machine_buying_cost': machine_buying_cost_comp,
                'total_machine_removal_cost': machine_removal_cost_comp,
                'total_inventory_holding_cost': inventory_holding_cost_comp,
                'total_storage_level_fixed_cost': storage_level_fixed_cost_comp
            }

        ga_optimal_details = get_ga_optimal_plan_details(best_individual)

        results_df_ga = pd.DataFrame({
            'Period': list(T),
            'Demand': [demand[t] for t in T],
            'Production': [ga_optimal_details['production'][t] for t in T],
            'Inventory': [ga_optimal_details['inventory'][t] for t in T],
            'Active Machines': [ga_optimal_details['active_machines'][t] for t in T],
            'Machines Bought': [ga_optimal_details['machines_bought'][t] for t in T],
            'Machines Removed': [ga_optimal_details['machines_removed'][t] for t in T]
        }).set_index('Period')

        storage_df_ga = pd.DataFrame(ga_optimal_details['storage_choices']).set_index('Period')
        results_df_ga = results_df_ga.merge(storage_df_ga, left_index=True, right_index=True)

        print("\n--- Optimal Plan (Genetic Algorithm) ---")
        print(results_df_ga)

        print("\n--- Cost Analysis (Genetic Algorithm) ---")
        print(f"Optimal Total Cost (excluding penalties and max storage weight): {ga_optimal_details['total_costs_ga_actual']:.2f}")
        print(f"Final Maximum Storage Level: {ga_optimal_details['max_inventory_observed']:.2f}")
        print("\nCost Components:")
        print(f"  Production Cost: {ga_optimal_details['total_production_cost']:.2f}")
        print(f"  Machine Buying Cost: {ga_optimal_details['total_machine_buying_cost']:.2f}")
        print(f"  Machine Removal Cost: {ga_optimal_details['total_machine_removal_cost']:.2f}")
        print(f"  Inventory Holding Cost: {ga_optimal_details['total_inventory_holding_cost']:.2f}")
        print(f"  Storage Level Fixed Cost: {ga_optimal_details['total_storage_level_fixed_cost']:.2f}")

        # --- 7. Visualize GA Optimal Plan ---
        if display_graphs == 1:
            sns.set_style("whitegrid")
            fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 18), sharex=True)

            axes[0].plot(results_df_ga.index, results_df_ga['Inventory'], label='Inventory', color='blue', marker='o', linestyle='-')
            demand_series = pd.Series(demand)
            for i, period in enumerate(results_df_ga.index):
                min_units = results_df_ga.loc[period, 'Min_Units']
                max_units = results_df_ga.loc[period, 'Max_Units']
                axes[0].fill_between([period - 0.4, period + 0.4], min_units, max_units, color='green', alpha=0.2, label='_nolegend_' if i > 0 else 'Storage Capacity Range')
            axes[0].set_ylabel('Units')
            axes[0].set_title('Inventory Levels and Storage Capacity Over Time (Genetic Algorithm)')
            axes[0].legend()

            axes[1].plot(results_df_ga.index, results_df_ga['Production'], label='Production', color='red', marker='x', linestyle='--')
            axes[1].plot(demand_series.index, demand_series.values, label='Demand', color='purple', marker='^', linestyle=':')
            axes[1].set_ylabel('Units')
            axes[1].set_title('Production vs. Demand Over Time (Genetic Algorithm)')
            axes[1].legend()

            axes[2].plot(results_df_ga.index, results_df_ga['Active Machines'], label='Active Machines', color='orange', marker='s', linestyle='-')
            axes[2].set_xlabel('Period')
            axes[2].set_ylabel('Number of Machines')
            axes[2].set_title('Active Machines Over Time (Genetic Algorithm)')
            axes[2].legend()

            plt.tight_layout()
            plt.show()

        cost_and_runtimedf = pd.DataFrame({
            'Production Cost': [ga_optimal_details['total_production_cost']],
            'Machine Buying Cost': [ga_optimal_details['total_machine_buying_cost']],
            'Machine Removal Cost': [ga_optimal_details['total_machine_removal_cost']],
            'Inventory Holding Cost': [ga_optimal_details['total_inventory_holding_cost']],
            'Storage Level Fixed Cost': [ga_optimal_details['total_storage_level_fixed_cost']],
            'max_inventory_observed': [ga_optimal_details['max_inventory_observed']],
            'ga_runtime_seconds': [ga_runtime]
        })

        return results_df_ga, cost_and_runtimedf

    except Exception as e:
        print(f"An unexpected error occurred during OptUsingGA for {jsonfilename}: {e}")
        traceback.print_exc() # Print full traceback
        return None, None

# --- Rerunning Example Usage for OptUsingGA with updated function ---
# Ensure drive is mounted if not already
drive.mount('/content/drive', force_remount=True)

test_data_dir = '/content/drive/MyDrive/Tesco_assignment/dec_2025/TestData'
results_data_dir = '/content/drive/MyDrive/Tesco_assignment/dec_2025/Results/'

runs_analysis_ga_lst = []

# Create results directory if it doesn't exist
os.makedirs(results_data_dir, exist_ok=True)

for file in os.listdir(test_data_dir):
    if file.endswith('.json'):
        print(f"Processing {file} with Genetic Algorithm")
        json_file_path = os.path.join(test_data_dir, file)
        jsonfilename = file.split('.')[0]

        results_df_ga, cost_runTime_ga = OptUsingGA(json_file_path, jsonfilename, display_graphs=1)

        if results_df_ga is not None and cost_runTime_ga is not None:
            cost_runTime_ga['Expt'] = jsonfilename
            runs_analysis_ga_lst.append(cost_runTime_ga)
            results_df_ga.to_csv(os.path.join(results_data_dir, f'{jsonfilename}_GA_results.csv'))
        else:
            print(f"Skipping results for {jsonfilename} due to an error during GA execution.")

if runs_analysis_ga_lst:
    all_runs_analysis_ga_df = pd.concat(runs_analysis_ga_lst, axis=0)
    all_runs_analysis_ga_df.to_csv(os.path.join(results_data_dir, 'all_runs_analysis_GA.csv'), index=False)
    print("\nGenetic Algorithm analysis for all JSONs saved to 'all_runs_analysis_GA.csv'.")
else:
    print("No JSON files were successfully processed by Genetic Algorithm to collect analysis.")
