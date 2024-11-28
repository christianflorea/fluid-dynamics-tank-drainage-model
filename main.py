import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sympy as sp

g = 9.81       
rho = 1000     
mu = 0.001     
epsilon = 0.0000025

A_tank = 0.26 * 0.32

D_pipe = 0.00794  
A_pipe = np.pi * (D_pipe / 2) ** 2 

h0 = 0.08 # init height
hf = 1e-6  # final height

def friction_factor(Re):
    if Re < 2000:
        f = 64 / Re
    else:
        f_symbol = sp.symbols('f')
        equation = 1 / sp.sqrt(f_symbol) + 2 * sp.log((epsilon / (3.7 * D_pipe)) + (2.51 / (Re * sp.sqrt(f_symbol))))
        f_initial_guess = 0.02
        try:
            f_solution = sp.nsolve(equation, f_symbol, f_initial_guess, verify=False)
            f = float(abs(f_solution))
        except Exception as e:
            print(f"Could not solve for friction factor for Re={Re:.2e}, epsilon={epsilon}, D={D_pipe}. Error: {e}")
            f = f_initial_guess
    return f

# ODE
def dhdt(t, h, L):
    print(f'height = {h[0]}')
    h_current = max(h[0], 0.0) 
    if h_current <= 0:
        return [0.0]

    f = 0.0004

    max_iterations = 50
    tolerance = 1e-3
    for _ in range(max_iterations):
        denominator = D_pipe + 2 * (f * L)
        if denominator <= 0:
            return [0.0]
        v = np.sqrt((2 * g * D_pipe * h_current) / denominator)
        Re = (rho * v * L) / mu
        # print(f"t={t:.2f}s, h={h_current:.6f}m, Re={Re:.2e}, f={f:.6f}, v={(v * 1000):.6f}mm/s, L={L:.6f}m")
        f_new = friction_factor(Re)
        if np.abs(f_new - f) < tolerance:
            f = f_new
            break
        f = f_new
    else:
        print(f"Friction factor did not converge at t={t:.2f}s, h={h_current:.6f}m, Re={Re:.2e}")
    dh_dt = - (A_pipe / A_tank) * v
    return [dh_dt]

def event_h_zero(t, h, L):
    return h[0] - hf

event_h_zero.terminal = True 
event_h_zero.direction = -1  

tube_lengths_cm = [20, 30, 40, 60]
tube_lengths_m = [l / 100 for l in tube_lengths_cm]

drain_times = []

for L in tube_lengths_m:

    sol = solve_ivp(
        dhdt, [0, 10000], [h0],
        args=(L,),
        events=event_h_zero,
        dense_output=True,
        max_step=1,
        atol=1e-8,
        rtol=1e-8
    )
    if sol.t_events[0].size > 0:
        t_drain = sol.t_events[0][0]
        drain_times.append(t_drain)
        print(f"Tube Length: {L*100:.0f} cm, Drain Time: {t_drain/60:.2f} minutes")
    else:
        print(f"Tube Length: {L*100:.0f} cm, Drain Time could not be computed.")
        print(f"Solver status: {sol.status}, Message: {sol.message}")
        print(f"Final time: {sol.t[-1]:.2f}s, Final height: {sol.y[0, -1]:.6f}m")
        drain_times.append(np.nan)

exp_times = [(3*60+19)/60, (3*60+34)/60, (4*60+26)/60, (4*60+48)/60]

plt.figure(figsize=(8, 6))
plt.plot(tube_lengths_cm, np.array(drain_times)/60, marker='o', label='Model Prediction (No Minor Losses)')
plt.plot(tube_lengths_cm, exp_times, marker='s', label='Experimental Data')
plt.xlabel('Tube Length (cm)')
plt.ylabel('Drain Time (minutes)')
plt.title('Drain Time vs Tube Length')
plt.legend()
plt.grid(True)
plt.show()
