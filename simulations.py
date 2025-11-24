import numpy as np
import matplotlib.pyplot as plt #Would be great to make some functions to plot some things?
from numba import jit
import os
import datetime
import argparse

@jit(nopython=True)
def ornstein_uhlenbeck_steps_numba(n, deltaT, T, v0, alpha):
    """
    TODO
    """
    X = np.zeros(n, dtype=np.float64)
    sqrt_term = np.sqrt(2.0 / float(T))
    f_dt = float(deltaT)
    f_T = float(T)

    for i in range(n - 1):
      X[i+1] = X[i] + (-(X[i] / f_T) + sqrt_term * np.random.normal(0, 1)) * f_dt
    V = float(v0) * np.exp(float(alpha) * X)
    return V

@jit(nopython=True)
def simulate_rt_ou_vectorized_with_potential(
    n_traj, n_steps, v_run, dt, x0,
    T_ou, v0_ou, alpha_ou,
    alpha_force, omega0):
    """
    TODO
    """

    trajectories = np.zeros((n_traj, n_steps + 1), dtype=np.float64)
    x = np.full(n_traj, float(x0))
    trajectories[:, 0] = x
    X_ou = np.zeros(n_traj, dtype=np.float64)
    v = np.sign(np.random.rand(n_traj) - 0.5) * float(v_run)

    f_dt = float(dt)
    f_T_ou = float(T_ou)
    f_v0_ou = float(v0_ou)
    f_alpha_ou = float(alpha_ou)

    sqrt_term_ou = np.sqrt(2.0 * f_dt / f_T_ou)
    dt_over_T = f_dt / f_T_ou


    first_passage_times = np.full(n_traj, np.nan)
    escaped = np.zeros(n_traj, dtype=np.bool_)

    barrier_x = 0.0
    if alpha_force > 0:
        barrier_x = omega0 / (3.0 * alpha_force)

    frozen_positions = np.zeros(n_traj)

    for t in range(1, n_steps + 1):
        eta_ou = np.random.randn(n_traj)

        x_old = x.copy()

        X_ou = X_ou - (X_ou * dt_over_T) + (sqrt_term_ou * eta_ou)
        v_rate_t = f_v0_ou * np.exp(f_alpha_ou * X_ou)

        force_potential = -x_old * (omega0 - 3 * alpha_force * (np.abs(x_old)))

        x_new = x_old + v * f_dt + force_potential * f_dt

        p_tumble = v_rate_t * f_dt
        tumble_mask = np.random.rand(n_traj) < p_tumble
        v[tumble_mask] = -v[tumble_mask]

        current_time = t * f_dt

        still_running = np.isnan(first_passage_times) #particles that still didn't scape
        escaped_now = still_running & (np.abs(x_new) > barrier_x) #those that scape in this step
        first_passage_times[escaped_now] = current_time #update escape time
        escaped[escaped_now] = True #and the array wich tells who scaped and who don't
        frozen_positions[escaped_now] = np.sign(x_new[escaped_now]) * barrier_x
        # update positions
        x = np.where(
            escaped_now, #True or false
            frozen_positions, #if the particle escaped in this step, frozen position
            np.where(still_running, x_new, frozen_positions) #if it still tumbling, x_new, if it scaped before, frozen position
        )


        trajectories[:, t] = x

        final_time = n_steps * dt #final time of simulation
        nan_mask = np.isnan(first_passage_times) #those who did'nt escape
        first_passage_times[nan_mask] = final_time+dt #a time of escape greather than final
        #time tells you the particle never escaped. 

    return trajectories, first_passage_times, escaped

def run_simulation(**params
#        path, #Don't give a path, let the implementation infer it
):
    print("Runnning simulation")

    # Get the path to save from the folder where this script is located
    base_path = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_path, 'simulation_results/')
    os.makedirs(output_dir, exist_ok=True)

    #Create a unique name por each simulation
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f'simulation{timestamp}.npz'
    file_to_save = os.path.join(output_dir, file_name)

    #unpack params
    N_traj = params["N_traj"] #number of trajectories
    N_steps = params["N_steps"] #number of steps   
    V_run = params["V_run"] #scape velocity
    dt = params["dt"] #delta t
    x0 = params["x0"] #starting position
    T = params["T"] #memory time
    v0 = params["v0"] #initial tumbling rate
    alpha_tumbling = params["alpha_tumbling"] #the alpha parameter controlling the change in tumbling rate (ornstein-uhlenbeck) 
    alpha_potential = params["alpha_potential"] #the alpha parameter to tune the height of the barrier
    omega0 = params["omega0"] #the curvature of the potential

    #Simulation!! A single simulation, because ram issues
    #Maybe parallel can be used in the future? Or a function that calls this one several times
    trajectories, escape_times, escaped_particles = simulate_rt_ou_vectorized_with_potential(N_traj, N_steps, V_run, dt, x0, T, v0, alpha_tumbling, alpha_potential, omega0)
    msd_t = np.mean((trajectories - x0)**2, axis=0)
    time = np.arange(trajectories.shape[1]) * dt

    np.savez(file_to_save, 
#            trajectories = trajectories, #this was making the npz of huge size, don't save
            msd_t = msd_t,
            time = time,
            escape_times = escape_times,
            escaped_particles = escaped_particles,
            params_keys = np.array(list(params.keys())),
            params_values = np.array(list(params.values()))
    )
#            params=np.array([
#                    N_traj, N_steps, V_run, dt, x0,
#                     T, v0, alpha_tumbling, alpha_potential, omega0
#                                                                    ]))
    print(f"Saved to {file_to_save}")
    
def run_batch(param_name, values, **base_params):
    print(f"Running batch over {param_name}: {values}")

    for v in values:
        params = base_params.copy()
        params[param_name] = v
        print(f"\n--- Running {param_name} = {v} ---")
        run_simulation(**params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RT-OU simulations with potential barrier.")

    #default values --> the particles escape the potential with enough steps
    parser.add_argument("--N_traj", type=int, default=250)
    parser.add_argument("--N_steps", type=int, default=500000)
    parser.add_argument("--V_run", type=float, default=8.5)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--x0", type=float, default=0.0)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--v0", type=float, default=0.1)
    parser.add_argument("--alpha_tumbling", type=float, default=0.5)
    parser.add_argument("--alpha_potential", type=float, default=0.01)
    parser.add_argument("--omega0", type=float, default=1.0)

    #batch parameters (vary one param at a time)
    parser.add_argument("--batch", nargs="+", help="Parameter sweep: param value1 value2 ...")

    args = parser.parse_args()
    params = vars(args)


    # if batch mode
    if args.batch is not None:
        param_to_vary = args.batch[0]
        values = list(map(float, args.batch[1:]))

        # remove batch key from base params to pass it to run_batch
        params.pop("batch")

        #call run_batch with the param to vary, the values to simulate, and the rest of the 
        #params
        run_batch(param_to_vary, values, **params)

    # else, single simulation
    else:
        run_simulation(**params)