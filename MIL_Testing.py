import pybamm
import numpy as np

def pybammsimulation(current_time, step_size, current):
    if not hasattr(pybammsimulation, "model"):
        options = {"thermal": "lumped"}
        pybammsimulation.model = pybamm.lithium_ion.DFN(options)
        pybammsimulation.parameter_values = pybamm.ParameterValues("Chen2020")
        pybammsimulation.parameter_values.update({"Ambient temperature [K]": 298.15})
        pybammsimulation.parameter_values.update({"Electrolyte conductivity [S.m-1]": 0.07})
        pybammsimulation.parameter_values.update({"Upper voltage cut-off [V]": 5.2})
        pybammsimulation.parameter_values.update({"Lower voltage cut-off [V]": 3.0})
        pybammsimulation.simulation = None
        pybammsimulation.solution = None
        pybammsimulation.sim_time = 0
        pybammsimulation.ini_soc = 0.9

    if pybammsimulation.simulation is None:
        pybammsimulation.simulation = pybamm.Simulation(
            pybammsimulation.model,
            parameter_values=pybammsimulation.parameter_values,
            solver=pybamm.CasadiSolver(mode="safe", rtol=1e-6, atol=1e-8)
        )
        t_eval = np.linspace(pybammsimulation.sim_time, pybammsimulation.sim_time + step_size, 10)
        pybammsimulation.solution = pybammsimulation.simulation.solve(
            t_eval=t_eval
        )
        pybammsimulation.sim_time += step_size

    pybammsimulation.parameter_values.update({"Current function [A]": current})
    pybammsimulation.model = pybammsimulation.model.set_initial_conditions_from(
        pybammsimulation.solution, inplace=False
    )
    pybammsimulation.simulation = pybamm.Simulation(
        pybammsimulation.model,
        parameter_values=pybammsimulation.parameter_values,
        solver=pybamm.CasadiSolver(mode="safe", rtol=1e-6, atol=1e-8)
    )

    t_eval = np.linspace(current_time, current_time + step_size, 100)
    pybammsimulation.solution = pybammsimulation.simulation.solve(t_eval=t_eval)
    pybammsimulation.sim_time = current_time + step_size

    current_avg = np.mean(pybammsimulation.solution["Current [A]"].entries)
    discharged_capacity = current_avg * (step_size / 3600)  
    pybammsimulation.ini_soc -= discharged_capacity / pybammsimulation.parameter_values["Nominal cell capacity [A.h]"]
    pybammsimulation.ini_soc = max(0.0, min(1.0, pybammsimulation.ini_soc))
    voltages = float(np.mean(pybammsimulation.solution["Terminal voltage [V]"].entries))
    print(f"Time: {current_time}s, Current: {current_avg:.3f} A, Voltage: {voltages:.3f} V, SOC: {pybammsimulation.ini_soc:.3f}")
    return voltages, pybammsimulation.ini_soc

if __name__ == "__main__":
    step_size = 1
    total_time = 3600
    current = 5

    print("Initializing simulation...")
    for current_time in range(0, total_time + step_size, step_size):
        print(f"Running simulation for time step: {current_time}")
        voltages, soc = pybammsimulation(current_time, step_size, current)
        print(f"Voltage at time {current_time}: {voltages}")
        print(f"SOC at time {current_time}: {soc}")