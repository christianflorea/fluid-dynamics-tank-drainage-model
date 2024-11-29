import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ODE : dh/dt = - (A_pipe / A_tank) * v
# v = sqrt((2 * g * h) / (1 + (f * L / D_pipe) + K_total))
# f_laminar = 64 / Re
# f_turbulent = 1 / (-2 * log10((epsilon / (3.7 * D_pipe)) + (2.51 / (Re * sqrt(f))))^2)

# Constants and parameters
g = 9.81            # (m/s^2)
rho = 1000          # (kg/m^3)
mu = 1e-3           # (Pa·s)
epsilon = 0.0000025 # (m)

A_tank = 0.26 * 0.32    # (m^2)
D_pipe = 0.00794        # (m)
A_pipe = np.pi * (D_pipe / 2) ** 2  # (m^2)

h0 = 0.12  # (m)
hf = 0.04  # (m)

# Minor losses
K_entry = 0.5
K_total = K_entry

def friction_factor(Re):
    if Re < 2000:
        # Laminar flow
        f = 64 / Re
    else:
        # Turbulent flow
        f_initial_guess = 0.01
        f = f_initial_guess
        max_iterations = 50
        tolerance = 1e-6
        for _ in range(max_iterations):
            rhs = -2.0 * np.log10((epsilon / (3.7 * D_pipe)) + (2.51 / (Re * np.sqrt(f))))
            f_new = (1 / rhs) ** 2
            if abs(f_new - f) < tolerance:
                f = f_new
                break
            f = f_new
        else:
            print(f"Friction factor did not converge for Re={Re:.2e}")
    return f

vel = []
time = []

def dhdt(t, h, L):
    h_current = max(h[0], 0.0)
    if h_current <= 0:
        return [0.0]

    f = 0.01
    max_iterations = 50
    tolerance = 1e-6

    for _ in range(max_iterations):
        v = np.sqrt((2 * g * h_current) / (1 + (f * L / D_pipe) + K_total))
        Re = (rho * v * D_pipe) / mu
        f_new = friction_factor(Re)
        if np.abs(f_new - f) < tolerance:
            f = f_new
            break
        f = f_new
    else:
        print(f"Friction factor did not converge at t={t:.2f}s, h={h_current:.6f} m, Re={Re:.2e}")
    dh_dt = - (A_pipe / A_tank) * v

    if L == 0.2:
        vel.append(v)
        time.append(t)

    return [dh_dt]

def event_h_zero(t, h, L):
    return h[0] - hf

event_h_zero.terminal = True
event_h_zero.direction = -1

# Tube lengths from 10 cm to 490 cm
tube_lengths_cm = [x * 10 for x in range(1, 50)]

# Tube lengths from experimental data
# tube_lengths_cm = [20, 30, 40, 60]
tube_lengths_m = [l / 100 for l in tube_lengths_cm]

drain_times = []
exp_times_cm = [20, 30, 40, 60]
exp_times = [(3*60+19)/60, (3*60+34)/60, (4*60+26)/60, (4*60+48)/60]

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
        print(f"Tube Length: {L*100:.0f} cm, Drain Time: {int(t_drain // 60)} mins {int(t_drain % 60)} secs")
    else:
        print(f"Tube Length: {L*100:.0f} cm, Drain Time could not be computed.")
        drain_times.append(np.nan)

# Convert drain times to minutes for plotting
drain_times_minutes = np.array(drain_times) / 60

# Plotting drain time vs tube length
plt.figure(figsize=(8, 6))
plt.plot(tube_lengths_cm, drain_times_minutes, marker='o', label='Model Prediction')
plt.plot(exp_times_cm, exp_times, marker='s', label='Experimental Data')
plt.xlabel('Tube Length (cm)')
plt.ylabel('Drain Time (minutes)')
plt.title('Drain Time vs Tube Length')
plt.legend()
plt.grid(True)
plt.show()

# Plotting velocity vs time for L = 20 cm
plt.figure(figsize=(8, 6))
plt.plot(time, vel)
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity vs Time for Tube Length = 20 cm')
plt.grid(True)
plt.show()

# Plotting time-to-length ratio vs tube length
time_length_ratio = [drain_times[i] / tube_lengths_m[i] for i in range(len(tube_lengths_m))]
plt.figure(figsize=(8, 6))
plt.plot(tube_lengths_cm, time_length_ratio, marker='o')
plt.xlabel('Tube Length (cm)')
plt.ylabel('Time-to-Length Ratio (s/m)')
plt.title('Time-to-Length Ratio vs Tube Length')
plt.grid(True)
plt.show()
