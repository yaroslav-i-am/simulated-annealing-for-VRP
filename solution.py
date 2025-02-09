import os
import re
from typing import Any
import numpy as np
import vrplib
from pprint import pprint
import random
from tqdm import trange
import matplotlib.pyplot as plt 
from copy import deepcopy

class State:
    def __init__(self, n_cars: int = None, n_nodes: int = None, route: list[int] = None, demand: list[int] = None, capacity: int = None, verbose: bool = False):

        self.n_cars = n_cars
        self.n_nodes = n_nodes
        self.demand = demand.tolist()[1:]
        self.capacity = capacity
        self.verbose = verbose

        self.resulting_routes = {i: [1] for i in range(1, n_cars + 1)}
        self.resulting_demands = {i: [0] for i in range(1, n_cars + 1)}

        cars_free_space = {i: self.capacity for i in range(1, n_cars + 1)}

        _nodes_iter = list(range(2, n_nodes + 1))
        random.shuffle(_nodes_iter)

        for node in _nodes_iter:
            node_demand = self.demand[node - 2]
            if verbose:
                print(f"Node: {node}. Demand = {node_demand}.")

            # print(f"{cars_free_space = }")

            try:
                car_index, _ = min(
                    filter(lambda el: el[1] >= node_demand, cars_free_space.items()), 
                    key=lambda el: el[1]
                )
            except ValueError:
                print("Unable to split nodes by default. Retrying...")
                self.__init__(n_cars, n_nodes, demand=demand, capacity=capacity, verbose=False)
                return

            cars_free_space[car_index] -= node_demand
            self.resulting_routes[car_index].append(node)
            self.resulting_demands[car_index].append(node_demand)

        for car_index, resulting_demand in self.resulting_demands.items():
            assert sum(resulting_demand) <= self.capacity

            self.resulting_routes[car_index].append(1)
            self.resulting_demands[car_index].append(0)


    def change(self):

        exit_ok = False

        again = 0

        file = open("stats.txt", "+at")

        while not exit_ok:
            if again == 0 or again > 20:
                mode = random.randint(1, 3) #  random.choice([1, 3])

            if self.verbose:
                print(f"Attempting {mode}, {again=}")

            car1, car2, elem1, elem2 = [None] * 4

            if mode == 1:  # Перестановка 2х точек разных маршрутов.                

                car1 = random.randint(1, self.n_cars)
                car2 = random.randint(1, self.n_cars)
                
                if len(self.resulting_routes[car1]) < 3 or len(self.resulting_routes[car2]) < 3:
                    continue

                elem1 = random.randint(2, len(self.resulting_routes[car1]) - 1)
                elem2 = random.randint(2, len(self.resulting_routes[car2]) - 1)

                if car1 != car2:
                    exit_ok = True
                else:
                    again += 1
                    continue

                self.resulting_routes[car1][elem1 - 1], self.resulting_routes[car2][elem2 - 1] = self.resulting_routes[car2][elem2 - 1], self.resulting_routes[car1][elem1 - 1]
                self.resulting_demands[car1][elem1 - 1], self.resulting_demands[car2][elem2 - 1] = self.resulting_demands[car2][elem2 - 1], self.resulting_demands[car1][elem1 - 1]

                if sum(self.resulting_demands[car1]) > self.capacity or sum(self.resulting_demands[car2]) > self.capacity:
                    self.resulting_routes[car1][elem1 - 1], self.resulting_routes[car2][elem2 - 1] = self.resulting_routes[car2][elem2 - 1], self.resulting_routes[car1][elem1 - 1]
                    self.resulting_demands[car1][elem1 - 1], self.resulting_demands[car2][elem2 - 1] = self.resulting_demands[car2][elem2 - 1], self.resulting_demands[car1][elem1 - 1]
                    exit_ok = False
                    again += 1
                    continue

                file.write("1")


            elif mode == 2:  # Перестановка 2х точек одного маршрута.
                car1 = random.randint(1, self.n_cars)

                if len(self.resulting_routes[car1]) < 4:
                    continue

                elem1 = random.randint(2, len(self.resulting_routes[car1]) - 1)
                elem2 = random.randint(2, len(self.resulting_routes[car1]) - 1)

                if elem1 != elem2:
                    exit_ok = True
                else:
                    again += 1
                    continue

                self.resulting_routes[car1][elem1 - 1], self.resulting_routes[car1][elem2 - 1] = self.resulting_routes[car1][elem2 - 1], self.resulting_routes[car1][elem1 - 1]
                self.resulting_demands[car1][elem1 - 1], self.resulting_demands[car1][elem2 - 1] = self.resulting_demands[car1][elem2 - 1], self.resulting_demands[car1][elem1 - 1]

                file.write("2")

            elif mode == 3:  # Перестановка из одного маршрута в другой.
                car1 = random.randint(1, self.n_cars)
                car2 = random.randint(1, self.n_cars)

                if len(self.resulting_routes[car1]) < 4:
                    continue

                elem1 = random.randint(2, len(self.resulting_routes[car1]) - 1)
                elem2 = random.randint(2, len(self.resulting_routes[car2]))  # Поскольку на текущее место мы вставляем.


                if car1 != car2 and \
                    sum(self.resulting_demands[car2]) + self.resulting_demands[car1][elem1 - 1] <= self.capacity and \
                    len(self.resulting_routes[car1]) >= 4:
                    exit_ok = True
                else:
                    again += 1
                    continue

                self.resulting_routes[car2].insert(elem2 - 1, self.resulting_routes[car1][elem1 - 1])
                self.resulting_demands[car2].insert(elem2 - 1, self.resulting_demands[car1][elem1 - 1])

                self.resulting_routes[car1].pop(elem1 - 1)
                self.resulting_demands[car1].pop(elem1 - 1)

                file.write("3")


        if self.verbose:
            print(f"=== Change: {mode = } ===")
            print(f"{car1 = }, {elem1 = }, {car2 = }, {elem2 = }")

        for _, resulting_route in self.resulting_routes.items():
            assert resulting_route[0] == 1
            assert resulting_route[-1] == 1

        for _, resulting_demand in self.resulting_demands.items():
            assert resulting_demand[0] == 0
            assert resulting_demand[-1] == 0

        file.close()

        return self
    

    def __str__(self):
        return f"Routes: {self.resulting_routes}. \nDemands: {self.resulting_demands}."
    

    def __repr__(self):
        return self.__str__()
    

def loss_function(state: State, edge_weights: dict[str, Any]):
    loss = 0.0

    # print(f"{state.n_cars = }")
    for route in state.resulting_routes.values():
        for j in range(len(route) - 1):
            loss += edge_weights[route[j] - 1][route[j + 1] - 1]
    
    return loss


def run_simulated_annealing(
        n_cars: int, 
        n_nodes: int, 
        demand: list[int], 
        capacity: int, 
        edge_weights: np.ndarray,
        prob_multiplier: float = 0.2,
        alpha: float = 0.99994,
        initial_temp = 100,
        existing_state = None
    ):
    N_ITERATIONS = 100_000
    INITIAL_TEMPERATURE = initial_temp

    if existing_state:
        initial_state = existing_state
    else:
        initial_state = State(n_cars, n_nodes, demand=demand, capacity=capacity, verbose=False)

    temp = INITIAL_TEMPERATURE
    state = initial_state
    state_losses = [loss_function(state, edge_weights)]
    temperatures = []
    probabilities = []
    stds = []

    best_state = None
    best_loss = None

    for iter_n in trange(N_ITERATIONS, leave=False):
        potential_state = deepcopy(state)
        potential_state.change()
        
        temperatures.append(temp)

        new_loss = loss_function(potential_state, edge_weights)
        old_loss = loss_function(state, edge_weights)
        state_losses.append(new_loss)

        energy_delta = new_loss - old_loss

        if energy_delta < 0:
            # print(f"{energy_delta = }, {temp = }, {np.exp(-energy_delta / temp) = }")
            state = potential_state

            best_state = state
            best_loss = state_losses[-1]
        else:
            probability = np.exp(-energy_delta / temp) # OK = 0.2
            probabilities.append(probability)

            _number = np.random.rand()

            if _number < probability:
                state = potential_state
                state_losses.append(new_loss)

                best_state = state

        # temp *= alpha
        # std = np.std(state_losses[-1000:])
        # stds.append(std)
        # # if len(state_losses) >= 1000 and std < 50:
        # if len(state_losses) >= 20000 and np.min(state_losses[:-10000]) < np.min(state_losses[-10000:]):
        #     temp = initial_temp
        # else:

        temp = max(temp / (1 + alpha * np.log(1 + iter_n)), 0)

    # plt.plot(stds)

    return best_state, best_loss, state_losses, temperatures, probabilities
    


if __name__ == "__main__":
    losses = []
    targets = []
    for path in filter(lambda p: p.endswith(".vrp"), os.listdir("./Vrp-Set-A/")[1:]):
        instance = vrplib.read_instance(f"./Vrp-Set-A/A-n32-k5.vrp")
        optimal_value = float(re.findall(r"Optimal value: \d+", instance["comment"])[0].split()[-1])

        pprint(instance)

        n_nodes, n_cars = map(int, re.findall(r"\d+", instance["name"]))
        print(f"{n_nodes = }, {n_cars = }")

        test_state = State(n_cars, n_nodes, None, instance["demand"], instance["capacity"])
        print(test_state)

        new_test_state = deepcopy(test_state)

        try:
            new_test_state.change()
        except Exception as e:
            print(f"Exception!")
            raise
        finally:
            print(new_test_state)

        random.seed(911)
        best_state, best_loss, state_losses, temperatures, probabilities = run_simulated_annealing(n_cars, n_nodes, instance["demand"], instance["capacity"], instance["edge_weight"])

        losses.append(best_loss)
        targets.append(optimal_value)

        print(f"{best_loss = } | {optimal_value = }")

        # current_state = State(n_cars, n_nodes, demand=instance["demand"], capacity=instance["capacity"])
        # print("State:")
        # print(current_state)
        # print(current_state.change())

        break

# plt.plot(state_losses)
# plt.show()

# plt.plot(temperatures)
# plt.show()

# plt.hist(probabilities, bins=100)
# plt.show()