#############Single Stage Code--Need to run for multiple input files and generate time comparison---######
### store number of variables created and runtime required######
import pulp
from google.colab import drive
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

def OptUsingMIP(json_file_path,jsonfilename,display_graphs=1):
  try:
      with open(json_file_path, 'r') as f:
          json_data_dict = json.load(f)
      print(f'Successfully loaded "{json_file_path}" into a Python dictionary.')
  except FileNotFoundError:
      print(f'Error: The file "{json_file_path}" was not found.')
      exit()
  except json.JSONDecodeError as e:
      print(f'Error: Could not decode JSON from "{json_file_path}". Error: {e}')
      exit()
  except Exception as e:
      print(f'An unexpected error occurred during JSON loading: {e}')
      exit()

  # Create DataFrame for storage details
  df_storage_details = pd.DataFrame({
      "Storage_levels": json_data_dict["Storage_levels"],
      "Minimum_units_per_storage_level": json_data_dict["Minimum_units_per_storage_level"],
      "Maximum_units_per_storage_level": json_data_dict["Maximum_units_per_storage_level"],
      "Cost_per_storage_level": json_data_dict["Cost_per_storage_level"]
  })
  print(df_storage_details.head())

  # Create DataFrame for periods, demand, and production rates
  df_periods_demand_products = pd.DataFrame({
      "Period": json_data_dict["Periods"],
      "Demand_per_period": json_data_dict["Demand_per_period"],
      "Products_units_per_machine_per_period": json_data_dict["Products_units_per_machine_per_period"]
  })
  print(df_periods_demand_products.head())

  # Create dictionary for financial and machine details
  financial_and_machine_details_dict = {
      'Initial_inventory': json_data_dict['Initial_inventory'],
      'Initial_machine_number': json_data_dict['Initial_machine_number'],
      'Maximum_number_of_machines': json_data_dict['Maximum_number_of_machines'],
      'Cost_to_buy_machine': json_data_dict['Cost_to_buy_machine'],
      'Cost_to_remove_machine': json_data_dict['Cost_to_remove_machine'],
      'Cost_to_produce_one_unit_of_product': json_data_dict['Cost_to_produce_one_unit_of_product']
  }
  print(financial_and_machine_details_dict)



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


  print("Extracted parameters and created data structures:")
  print(f"T (Periods): {T[:5]}...") # print first 5 to avoid long output
  print(f"S (Storage Levels): {S}")
  print(f"Demand (first 5 items): {list(demand.items())[:5]}")
  print(f"Production Rate (first 5 items): {list(production_rate.items())[:5]}")
  print(f"Min Storage Units (first 5 items): {list(min_storage_units.items())[:5]}")
  print(f"Max Storage Units (first 5 items): {list(max_storage_units.items())[:5]}")
  print(f"Initial Inventory: {initial_inventory}")
  print(f"Initial Machines: {initial_machines}")
  print(f"Max Machines: {max_machines}")
  print(f"Cost to Buy Machine: {cost_buy_machine}")
  print(f"Cost to Remove Machine: {cost_remove_machine}")
  print(f"Cost to Produce Unit: {cost_produce_unit}")
  print(f"Cost to Store Unit per Period: {cost_store_unit_per_period}")
  print(f"Cost per Storage Level: {cost_storage_level}")


  # 1. Initialize a PuLP problem named model_single_stage with the objective to minimize
  model_single_stage = pulp.LpProblem("Single_Stage_Optimization", pulp.LpMinimize)

  # Define a large weight to prioritize minimizing max_storage_level_units
  WEIGHT = 1000

  # 2. Define the following PuLP variables:
  # P[t] for production quantity in each period t in T
  P_ss = pulp.LpVariable.dicts("Production_SS", T, lowBound=0, cat='Continuous')

  # I[t] for inventory level at the end of each period t in T
  I_ss = pulp.LpVariable.dicts("Inventory_SS", T, lowBound=0, cat='Continuous')

  # M[t] for the number of active machines in each period t in T
  M_ss = pulp.LpVariable.dicts("ActiveMachines_SS", T, lowBound=0, cat='Integer')

  # Mb[t] for the number of machines bought in each period t in T
  Mb_ss = pulp.LpVariable.dicts("MachinesBought_SS", T, lowBound=0, cat='Integer')

  # Mr[t] for the number of machines removed in each period t in T
  Mr_ss = pulp.LpVariable.dicts("MachinesRemoved_SS", T, lowBound=0, cat='Integer')

  # S_choice[t][s] for a binary variable (1 if storage level s is chosen in period t)
  S_choice_ss = pulp.LpVariable.dicts("StorageChoice_SS", (T, S), cat='Binary')

  # max_storage_level_units as an auxiliary variable for the maximum inventory level
  max_storage_level_units_ss = pulp.LpVariable("Max_Storage_Level_Units_SS", lowBound=0, cat='Continuous')

  # 3. Formulate and add the following constraints to model_single_stage:

  # Inventory Balance (Initial Period)
  model_single_stage += I_ss[T[0]] == initial_inventory + P_ss[T[0]] - demand[T[0]], "Inventory_Balance_Initial_SS"

  # Inventory Balance (Subsequent Periods)
  for t_idx in range(1, len(T)):
      t = T[t_idx]
      prev_t = T[t_idx - 1]
      model_single_stage += I_ss[t] == I_ss[prev_t] + P_ss[t] - demand[t], f"Inventory_Balance_SS_{t}"

  # Demand Fulfillment (I[t] >= 0 is handled by variable definition, implicit demand fulfillment)
  # Adding explicit constraint for clarity if needed, but lowBound=0 handles it.
  for t in T:
      model_single_stage += I_ss[t] >= 0, f"Demand_Fulfillment_SS_{t}"

  # Production Capacity
  for t in T:
      model_single_stage += P_ss[t] ==M_ss[t] * production_rate[t], f"Production_Capacity_SS_{t}"

  # Machine Dynamics (Initial Period)
  model_single_stage += M_ss[T[0]] == initial_machines + Mb_ss[T[0]] - Mr_ss[T[0]], "Machine_Dynamics_Initial_SS"

  # Machine Dynamics (Subsequent Periods)
  for t_idx in range(1, len(T)):
      t = T[t_idx]
      prev_t = T[t_idx - 1]
      model_single_stage += M_ss[t] == M_ss[prev_t] + Mb_ss[t] - Mr_ss[t], f"Machine_Dynamics_SS_{t}"

  # Max Machines Limit
  for t in T:
      model_single_stage += M_ss[t] <= max_machines, f"Max_Machines_Limit_SS_{t}"

  # Select One Storage Level
  for t in T:
      model_single_stage += pulp.lpSum(S_choice_ss[t][s] for s in S) == 1, f"Select_One_Storage_Level_SS_{t}"

  # Minimum Storage Units Constraint
  for t in T:
      model_single_stage += I_ss[t] >= pulp.lpSum(S_choice_ss[t][s] * min_storage_units[s] for s in S), f"Min_Storage_Units_SS_{t}"

  # Maximum Storage Units Constraint
  for t in T:
      model_single_stage += I_ss[t] <= pulp.lpSum(S_choice_ss[t][s] * max_storage_units[s] for s in S), f"Max_Storage_Units_SS_{t}"

  # Track Max Storage Level
  for t in T:
      model_single_stage += max_storage_level_units_ss >= I_ss[t], f"Track_Max_Storage_Level_SS_{t}"

  # 4. Define the objective function for model_single_stage
  production_costs_ss = pulp.lpSum(cost_produce_unit * P_ss[t] for t in T)
  machine_buying_costs_ss = pulp.lpSum(cost_buy_machine * Mb_ss[t] for t in T)
  machine_removal_costs_ss = pulp.lpSum(cost_remove_machine * Mr_ss[t] for t in T)
  inventory_holding_costs_ss = pulp.lpSum(cost_store_unit_per_period * I_ss[t] for t in T)
  storage_level_fixed_costs_ss = pulp.lpSum(S_choice_ss[t][s] * cost_storage_level[s] for t in T for s in S)

  model_single_stage += (production_costs_ss + machine_buying_costs_ss + machine_removal_costs_ss +
                        inventory_holding_costs_ss + storage_level_fixed_costs_ss) + \
                        (WEIGHT * max_storage_level_units_ss), "Total_Weighted_Costs_SS"

  # 5. Solve model_single_stage using pulp.HiGHS_CMD()
  try:
      start_time=time.time()
      status_ss = model_single_stage.solve(pulp.HiGHS_CMD(msg=True))
      end_time=time.time()
      print(f"Time taken to solve using Highs: {end_time-start_time} seconds")  
      timestatsdf=pd.DataFrame({"Solver":["Highs"],"Time":[end_time-start_time]})
      #timestatsdf.to_csv(f'TimeStat_{jsonfilename}.csv')

  except pulp.PulpSolverError:
      print("HiGHS solver not found or failed for Single Stage, falling back to CBC.")
      start_time=time.time()
      status_ss = model_single_stage.solve(pulp.PULP_CBC_CMD(msg=True))
      end_time=time.time()
      timestatsdf=pd.DataFrame({"Solver":["CBC"],"Time":[end_time-start_time]})
      #timestatsdf.to_csv(f'TimeStat_{jsonfilename}.csv')
      print(f"Time taken to solve using CBC: {end_time-start_time} seconds")  

  # 6. Print the status of the solution
  print(f"\nStatus (Single Stage): {pulp.LpStatus[status_ss]}")

  # 7. If an optimal solution is found, print the optimal value
  if pulp.LpStatus[status_ss] == 'Optimal':
      optimal_total_weighted_cost_ss = pulp.value(model_single_stage.objective)
      optimal_max_storage_level_ss = pulp.value(max_storage_level_units_ss)
      print(f"Optimal Total Weighted Cost (Single Stage): {optimal_total_weighted_cost_ss:.2f}")
      print(f"Optimal Maximum Storage Level (Single Stage): {optimal_max_storage_level_ss:.2f}")
  else:
      print("No optimal solution found for Single Stage.")

  if pulp.LpStatus[status_ss] == 'Optimal':
    # 1. Extract optimal values for all decision variables
    production_values_ss = {t: P_ss[t].varValue for t in T}
    inventory_values_ss = {t: I_ss[t].varValue for t in T}
    machine_values_ss = {t: M_ss[t].varValue for t in T}
    machine_bought_values_ss = {t: Mb_ss[t].varValue for t in T}
    machine_removed_values_ss = {t: Mr_ss[t].varValue for t in T}

    storage_choice_data_ss = []
    for t in T:
        chosen_s = None
        for s in S:
            if S_choice_ss[t][s].varValue == 1:
                chosen_s = s
                break
        storage_choice_data_ss.append({
            'Period': t,
            'Storage_Level': chosen_s,
            'Min_Units': min_storage_units[chosen_s],
            'Max_Units': max_storage_units[chosen_s],
            'Storage_Cost': cost_storage_level[chosen_s]
        })



    # 2. Store optimal values into a pandas DataFrame
    results_df_ss = pd.DataFrame({
        'Period': list(T),
        'Demand': [demand[t] for t in T],
        'Products_available':[production_values_ss[1]]+[production_values_ss[t]+inventory_values_ss[t-1] for t in T[1:]],
        'Production': [production_values_ss[t] for t in T],
        'Inventory': [inventory_values_ss[t] for t in T],
        'Active Machines': [machine_values_ss[t] for t in T],
        'Machines Bought': [machine_bought_values_ss[t] for t in T],
        'Machines Removed': [machine_removed_values_ss[t] for t in T]
    }).set_index('Period')

    storage_df_ss = pd.DataFrame(storage_choice_data_ss).set_index('Period')

    # Merge with main results_df_ss
    results_df_ss = results_df_ss.merge(storage_df_ss, left_index=True, right_index=True)
    #results_df_ss1=results_df_ss.drop(columns=[])
    #results_df_ss.to_csv(f'{jsonfilename}.csv')
    print("\n--- Optimal Plan (Single-Stage PuLP) ---")
    display(results_df_ss)

    # 3. Calculate total cost components
    total_production_cost_ss = sum(cost_produce_unit * production_values_ss[t] for t in T)
    total_machine_buying_cost_ss = sum(cost_buy_machine * machine_bought_values_ss[t] for t in T)
    total_machine_removal_cost_ss = sum(cost_remove_machine * machine_removed_values_ss[t] for t in T)
    total_inventory_holding_cost_ss = sum(cost_store_unit_per_period * inventory_values_ss[t] for t in T)
    total_storage_level_fixed_cost_ss = sum(storage_df_ss.loc[t, 'Storage_Cost'] for t in T)
    total_cost_df=pd.DataFrame({
        'total_production_cost_ss':[total_production_cost_ss],
        'total_machine_buying_cost_ss':[total_machine_buying_cost_ss],
        'total_machine_removal_cost_ss':[total_machine_removal_cost_ss],
        'total_inventory_holding_cost_ss':[total_inventory_holding_cost_ss],
        'total_storage_level_fixed_cost_ss':[total_storage_level_fixed_cost_ss],
        'objectiveValue':[optimal_total_weighted_cost_ss]}
    )
    # 4. Print the total minimized cost and cost breakdown
    # optimal_total_weighted_cost_ss and optimal_max_storage_level_ss are available from the previous step

    print("\n--- Cost Analysis (Single-Stage PuLP) ---")
    print(f"Optimal Total Weighted Cost: {optimal_total_weighted_cost_ss:.2f}")
    print(f"Optimal Maximum Storage Level: {optimal_max_storage_level_ss:.2f}")
    print("\nCost Components (Actual Costs, excluding weight for max storage):")
    print(f"  Production Cost: {total_production_cost_ss:.2f}")
    print(f"  Machine Buying Cost: {total_machine_buying_cost_ss:.2f}")
    print(f"  Machine Removal Cost: {total_machine_removal_cost_ss:.2f}")
    print(f"  Inventory Holding Cost: {total_inventory_holding_cost_ss:.2f}")
    print(f"  Storage Level Fixed Cost: {total_storage_level_fixed_cost_ss:.2f}")

    # Calculate the actual total cost (without the weight for max_storage_level_units) for comparison
    actual_total_cost_ss = total_production_cost_ss + total_machine_buying_cost_ss + \
                          total_machine_removal_cost_ss + total_inventory_holding_cost_ss + \
                          total_storage_level_fixed_cost_ss
    print(f"Actual Total Cost (excluding max storage weight): {actual_total_cost_ss:.2f}")

    if(display_graphs==1):
    # Set a style for better aesthetics
      sns.set_style("whitegrid")

      # Create a figure with 3 subplots stacked vertically
      fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 18), sharex=True)

      # Plot 1: Inventory Levels and Storage Capacity Over Time
      axes[0].plot(results_df_ss.index, results_df_ss['Inventory'], label='Inventory', color='blue', marker='o', linestyle='-')

      # Convert demand dictionary to a pandas Series for easy plotting (already done in previous steps)
      demand_series = pd.Series(demand)

      for i, period in enumerate(results_df_ss.index):
          min_units = results_df_ss.loc[period, 'Min_Units']
          max_units = results_df_ss.loc[period, 'Max_Units']
          # Plot fill_between for storage range. Use period values as x coordinates
          # The index starts from 1, so direct use of period is fine
          axes[0].fill_between([period - 0.4, period + 0.4], min_units, max_units, color='green', alpha=0.2, label='_nolegend_' if i > 0 else 'Storage Capacity Range')

      axes[0].set_ylabel('Units')
      axes[0].set_title('Inventory Levels and Storage Capacity Over Time (Single-Stage)')
      axes[0].legend()

      # Plot 2: Production vs. Demand Over Time
      axes[1].plot(results_df_ss.index, results_df_ss['Production'], label='Production', color='red', marker='x', linestyle='--')
      axes[1].plot(demand_series.index, demand_series.values, label='Demand', color='purple', marker='^', linestyle=':')
      axes[1].set_ylabel('Units')
      axes[1].set_title('Production vs. Demand Over Time (Single-Stage)')
      axes[1].legend()

      # Plot 3: Active Machines Over Time
      axes[2].plot(results_df_ss.index, results_df_ss['Active Machines'], label='Active Machines', color='orange', marker='s', linestyle='-')
      axes[2].set_xlabel('Period')
      axes[2].set_ylabel('Number of Machines')
      axes[2].set_title('Active Machines Over Time (Single-Stage)')
      axes[2].legend()

      # Adjust layout to prevent overlap
      plt.tight_layout()
      plt.show()
    results_df_ss.drop(columns=['Min_Units','Max_Units'],inplace=True)
    total_cost_df['Max Storage Level']=optimal_max_storage_level_ss
    cost_and_runtimedf=pd.concat([total_cost_df,timestatsdf],axis=1)  
    return (results_df_ss,cost_and_runtimedf)
  else:
    print("No optimal solution found for Single Stage.")
