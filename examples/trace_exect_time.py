"""tracé du temps. d'execution avec regression quadratique """

import csv
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict 
from scipy.optimize import curve_fit


def read_csv_data(filename="execution_times.csv"):
    """Read CSV file without pandas."""
    
    if not os.path.isfile(filename):
        return None
    
    data = defaultdict(list)
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        
        for row in reader:
            if not row or len(row) < 3:
                continue
            if not row[0] or not row[1] or not row[2]:
                continue
            
            try:
                N = int(row[0])
                backend = row[1]
                mean_time = float(row[2])
                data[(N, backend)].append(mean_time)
            except ValueError:
                continue
    
    if not data:
        return None
    
    return data, headers



def cout_funct(N, overhead, alpha1, alpha2,):
    return overhead + alpha1 * N + alpha2 * N**2
          


def plot_from_csv(filename="execution_times.csv"):
    """Plot log-log scaling from CSV file."""
    
    result = read_csv_data(filename)
    if result is None:
        return
    
    data, headers = result
    
    if not data:
        return
    
    avg ={}
    for (N, backend), times in data.items():
        avg_mean = np.mean(times)
        avg_std = np.std(times) if len(times) > 1 else 0.0
        avg[(N, backend)] = (avg_mean, avg_std)

    N_values = sorted({N for (N, _) in data.keys()})
    backends = sorted({backend for (_, backend) in data.keys()})
  

    
   
    results = {}
    
    for backend in backends:
        N_vect = np.array([])
        cout_vect = np.array([])
        ecart_type = np.array([])
        
        for N in N_values:
            if (N, backend) in avg:
                mean, std = avg[(N, backend)]
                N_vect = np.append(N_vect, N)
                cout_vect = np.append(cout_vect,mean)
                ecart_type= np.append(ecart_type, std)

        try:
            # Initialisation des paramÃ¨tres
            p0 = [0.0, 1.0, 0.0]  # overhead, alpha1, alpha2
            use_sigma =  np.all(ecart_type>0)

            if use_sigma:
                
            # Regression 
                popt, pcov = curve_fit(
                    cout_funct,         
                    N_vect,                
                    cout_vect,               
                    p0=p0,
                    bounds=([0,0,0],[np.inf, np.inf, np.inf]),              
                    sigma=ecart_type,        
                    absolute_sigma=True
                )
            else:
                print("ecart type nul")
                popt, pcov = curve_fit(
                    cout_funct,        
                    N_vect,                
                    cout_vect,                
                    p0=p0,                
                    bounds=([0,0,0],[np.inf, np.inf, np.inf]),
                )
            
            # Extraction des paramÃ¨tres
            overhead_opt, alpha1_opt, alpha2_opt = popt
            overhead_err, alpha1_err, alpha2_err = np.sqrt(np.diag(pcov))
            
            
            T_pred = cout_funct(N_vect, overhead_opt, alpha1_opt, alpha2_opt)
            residuals = cout_vect - T_pred
            rms = np.sqrt(np.mean(residuals**2))
            r2 = 1 - np.sum(residuals**2) / np.sum((cout_vect - np.mean(cout_vect))**2)
            
            
            results[backend] = {
                'overhead': overhead_opt,
                'alpha1': alpha1_opt,
                'alpha2': alpha2_opt,
                'overhead_err': overhead_err,
                'alpha1_err': alpha1_err,
                'alpha2_err': alpha2_err,
                'T_pred': T_pred,
                'residuals': residuals,
                'rms': rms,
                'r2': r2,
                'N_vect' : N_vect,
                'cout_vect' : cout_vect,
                'ecart_type' : ecart_type
            }
           
            
        except Exception as e:
            print(f"  âL ERREUR lors de la regression : {e}")

  
    print("resulats de la regression")

    print(f" {'backend':<12} {'overhead (s)':>15}  {'alpha1':>15} {'alpha2':>15}")

    
    
    # Premier graphique : log-log avec les fits

    backend_names = {
        "python" : "python",
        "cython" : "cython",
        "c" : "c",
        "cpp" : "cpp"

    }

    for backend in backends:
        if backend not in results:
            continue
            
        overhead = results[backend]['overhead']
        alpha1 = results[backend]['alpha1']
        alpha2 = results[backend]['alpha2']

    
        name = backend_names.get(backend, backend)
        print(f"{name:<12} {overhead:>15.6f} {alpha1:>15.2e} {alpha2:>15.2e}")
    

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = {"python": "blue", "cython": "orange", "c": "green", "cpp": "red"}
    markers = {"python": "o", "cython": "s", "c": "^", "cpp": "D"}
    labels = {"python": "Python", "cython": "Cython", "c": "C", "cpp": "C++"}

    
        
        
    for backend in backends:
        if backend == "python":
            continue
        if backend not in results:
            continue
            
        N_vect = results[backend]['N_vect']
        cout_vect = results[backend]['cout_vect']
        ecart_type = results[backend]['ecart_type']
        T_pred = results[backend]['T_pred']
    
    # Points expÃ©rimentaux
        ax1.errorbar(
            N_vect, cout_vect,
            yerr=ecart_type,
            marker=markers.get(backend, "o"),
            color=colors.get(backend, "black"),
            label=f"{labels.get(backend, backend)} (experimental)",
            linewidth=0,
            markersize=8,
            capsize=4,
            elinewidth=1,
        )
        """if first_mean_time is not None and len(N_values) > 0:
            N_ref = np.array(N_values)
            t_ref_n2 = N_ref**2 / (N_ref[0]**2 / first_mean_time)
            t_ref_n = N_ref / N_ref[0] * first_mean_time
            
            ax.loglog(N_ref, t_ref_n2, 'k--', label=r'$N^2$', alpha=0.7)
            ax.loglog(N_ref, t_ref_n, 'k:', label=r'$N$', alpha=0.7)"""
        
        # Courbe fit (points prÃ©dits)
        T_pred = results[backend]['T_pred']
        ax1.plot(N_vect, T_pred, 
                 color=colors.get(backend, "black"),
                 linestyle='-',
                 linewidth=2,
                 label=f"{labels.get(backend, backend)} (fit)")
    
    ax1.set_xlabel("Nombre de noeuds N")
    ax1.set_ylabel("Temps d'execution (s)")
    ax1.set_title("Temps d'exeution en fonction de N ")
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    
    for backend in backends:
        if backend == "python":
            continue
            
        N_vect = np.array([])
        for N in N_values:
            if (N, backend) in avg:
                N_vect = np.append(N_vect, N)
        
        residuals = results[backend]['residuals']
        ax2.scatter(N_vect, residuals, 
                   marker=markers.get(backend, "o"),
                   color=colors.get(backend, "black"),
                   label=labels.get(backend, backend),
                   s=50)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel("Nombre de noeuds N")
    ax2.set_ylabel("Residus (s)")
    ax2.set_title("residus du fit")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_from_csv("execution_times.csv")