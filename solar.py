from datetime import datetime
from pathlib import Path
from random import uniform
import re

from numba import njit
import pandas as pd

def read_solar_array_data(fp: Path) -> pd.DataFrame:
    raw = pd.read_csv(fp)
    raw['local_time'] = raw['LocalTime'].apply(lambda s: datetime.strptime(s, '%m/%d/%y %H:%M'))
    raw['power'] = raw['Power(MW)']
    raw = raw.drop(columns=['LocalTime', 'Power(MW)'])
    size = int(re.search(r'(\d+)MW', fp.stem)[1])
    raw['power'] /= size  # standardize on 1MW
    return raw


@njit
def uptime(capacity: float, load: float, sol: list[float]) -> tuple[float, float, float, float]:
    battery: list[float] = [capacity] + [0.0 for _ in sol] # MWh
    utilization: list[float] = [0.0 for _ in sol]  # percentage
    t_interval = 24.0 * 365.0 / len(sol)  # hours
    for i in range(len(sol)):
        sol_interval: float = sol[i]
        remaining_load: float = (load - sol_interval)*t_interval
        if sol_interval > load:  # More sun than load
            utilization[i] = 1  # Can run full load
            excess_solar = sol_interval - load
            # if battery[i] < capacity:
                # battery[i+1] = battery[i] + t_interval*excess_solar
            battery[i+1] = min(battery[i] + t_interval*excess_solar, capacity)
            
        elif battery[i] > remaining_load: # Battery is full enough
            utilization[i] = 1
            battery[i+1] = battery[i] - remaining_load
        else: # Battery fully drained, cannot use all load
            utilization[i] = (sol_interval * t_interval + battery[i]) / (load * t_interval)  # equivalent: sol_interval / load + battery / (load * ts)
            battery[i+1] = 0

    batsum = 0
    for b in battery[:-1]:
        batsum += 1 if b > 0 else 0
        
    return (
        capacity,
        load,
        batsum / (len(battery) - 1),
        t_interval * sum(utilization) / (24 * 365),
    )


@njit
def all_in_system_cost(
    solar_cost: float, battery_cost: float, load_cost: float, battery_size: float, array_size: float, sol: list[float]
) -> float:
    capacity, load, battery_util, total_util = uptime(
        capacity=battery_size / array_size,
        load=1 / array_size,
        sol=sol,
    )
    return (capacity * battery_cost + solar_cost + load_cost * load) / (load * total_util)


@njit
def cost_and_elasticity(
    solar_cost: float, battery_cost: float, load_cost: float, battery_size: float, array_size: float, sol: list[float]
) -> tuple[float, float, float, float, float]:
    size_to_cost = lambda b, a: all_in_system_cost(
        solar_cost, battery_cost, load_cost, b, a, sol
    )
    cost = size_to_cost(battery_size, array_size)
    cost_battery = size_to_cost(1.01*battery_size+0.01, array_size)
    cost_array = size_to_cost(battery_size, 1.01*array_size+0.01)
    return cost, cost_battery, cost_array, (cost - cost_battery) / cost, (cost - cost_array) / cost


@njit
def find_minimum_system_cost(
    solar_cost: float,
    battery_cost: float,
    load_cost: float,
    sol: list[float],
) -> tuple[tuple[float, float, float], tuple[float, float, 1], tuple[float, float, float], tuple[float, float, float], any]:
    bi = min(10, 10 * load_cost / 5e6)
    ai = min(10, 10 * load_cost / 5e6)
    amplitude = 10 + 70 * (load_cost / 5e6)
    if 7e5 < load_cost < 13e5: amplitude *= 3
    if 80e6 < load_cost: amplitude *= 0.5

    cost_min = 10**10
    bi_min = bi
    ai_min = ai
    for i in range(100):
        # There was a bug here: cost_and_elasticity were getting called with wrong argument order
        # Changing them around doesn't affect the results since we always passed in the same number here
        cost, cost_bat, cost_arr, dcost_bat, dcost_arr = cost_and_elasticity(solar_cost, battery_cost, load_cost, bi, ai, sol)
        if cost < cost_min:
            ai_min, bi_min, cost_min = ai, bi, cost
        # if True: print((cost, cost_bat, cost_arr, dcost_bat, dcost_arr), bi, ai)
        bi = max(0, bi + amplitude * uniform(0.1, 1) * dcost_bat)
        ai = max(0.01, ai + amplitude * uniform(0.1, 1) * dcost_arr)

    ut = uptime(bi_min / ai_min, 1 / ai_min, sol)
    
    array_cost = solar_cost * ai_min
    storage_cost = battery_cost * bi_min
    total_cost = array_cost + storage_cost + load_cost
    
    return (
        (solar_cost, battery_cost, load_cost),
        (ai_min, bi_min, 1),
        (array_cost, storage_cost, load_cost),
        (array_cost + storage_cost, total_cost, total_cost / ut[-1]),
        ut
    )