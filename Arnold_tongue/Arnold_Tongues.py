import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from NeuronNetworkSimulation import NeuronNetworkSimulation  


params_df = pd.read_csv('/Users/rodrigosanchez/Documents/Thesis /PINGpy/simulation_results.csv')
mean_contrasts = [30, 50, 70]  
delta_C_values = np.linspace(-20, 20, 10)  
coupling_strength_values = np.linspace(0.1, 1.0, 10)  


results = []

# load intercept and slope

for index, row in params_df.iterrows():
    intercept = row['Intercept']  
    slope = row['Slope']     

for C in mean_contrasts:
    for coupling_strength in coupling_strength_values:
        for delta_C in delta_C_values:
            
            contrast_net1 = C + delta_C / 2
            contrast_net2 = C - delta_C / 2
            
            input_net1 = (25 - intercept + 0.25 * contrast_net1) / slope
            input_net2 = (25 - intercept + 0.25 * contrast_net2) / slope

           
            simulation = NeuronNetworkSimulation(
                input_net1=input_net1,
                input_net2=input_net2,
                cop_val=coupling_strength,
                simulation_time=10000,  
                dt=1
            )
            simulation.run_simulation()

            
            signal1, signal2, px = simulation.analyze_results()
            plv = np.abs(np.mean(np.exp(1j * px)))  

           
            results.append({
                'mean_contrast': C,
                'coupling_strength': coupling_strength,
                'delta_C': delta_C,
                'plv': plv
            })


df_results = pd.DataFrame(results)
df_results.to_csv("arnold_tongue_results.csv", index=False)


for C in mean_contrasts:
    subset = df_results[df_results['mean_contrast'] == C]

    
    plv_matrix = subset.pivot(index='coupling_strength', columns='delta_C', values='plv')

    plt.figure(figsize=(8, 6))
    plt.imshow(plv_matrix, cmap='viridis', aspect='auto', origin='lower',
               extent=[delta_C_values.min(), delta_C_values.max(),
                       coupling_strength_values.min(), coupling_strength_values.max()])
    plt.colorbar(label='Phase-Locking Value (PLV)')
    plt.xlabel('Delta C (%)')
    plt.ylabel('Coupling Strength')
    plt.title(f'Arnold Tongue for Mean Contrast {C}%')
    plt.show()

