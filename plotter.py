import matplotlib.pyplot as plt

def plot(tube_lengths_cm, drain_times_minutes, exp_times_cm, exp_times, time, vel):
    '''
    Plotting functions
    '''
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
    time_length_ratio = [drain_times_minutes[i] / tube_lengths_cm[i] for i in range(len(tube_lengths_cm))]
    plt.figure(figsize=(8, 6))
    plt.plot(tube_lengths_cm, time_length_ratio, marker='o')
    plt.xlabel('Tube Length (cm)')
    plt.ylabel('Time-to-Length Ratio (s/m)')
    plt.title('Time-to-Length Ratio vs Tube Length')
    plt.grid(True)
    plt.show()