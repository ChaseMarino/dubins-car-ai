import numpy as np
import time
import random
import matplotlib.pyplot as plt
import math
from scipy import interpolate
from typing import List
from collections import namedtuple

individual = List[int]
stateHistory = namedtuple('state', ['cost','fitness', 'x','y','angle','velocity'])

#config 
maxGenerations = 500
maxPop = 201
time_to_park_car = 10
time_step = 0.1

mutationRate = .005
binaryParamSize = 7
k = 200
stopCriteria = 0.10
totalChromosomes = 10
minAccel = -5
maxAccel = 5
minHeading = -0.524
maxHeading = 0.524


all_time_steps_for_x = np.arange(0, time_to_park_car, time_step)
controlAmt = 2
total_entries = time_to_park_car * controlAmt

initialState =  [0, 8, 0, 0]
finalState = np.array([0, 0, 0, 0])

def newChromosome():
    res = np.random.randint(0,2, controlAmt * binaryParamSize * totalChromosomes).tolist()
    return res

#grey coding 
def graytoBinary(gray):
    res = [gray[0]]
    for i in range(1, len(gray)):
        res.append(not (res[i - 1]) if gray[i] else res[i - 1])
    return res

# binary conversion 
def toDecimal(binary):
    res = sum(b * 2**i for i, b in enumerate(reversed(binary)))
    return res if binary[0] else -res

def getVal(chromosome, range, min):
    oldVal = abs(toDecimal(chromosome))
    return ((oldVal * range) / (pow(2, binaryParamSize) -1)) + min

def getHeading(chromosome):
    return getVal(chromosome, maxHeading - minHeading, minHeading)

def getAcceleration(chromosome):
    return getVal(chromosome, maxAccel - minAccel, minAccel)

def calcCost(localFin, cf):
    if cf == 0:
        res = np.linalg.norm(localFin - finalState)
    else:
        res = k + cf
    return res

# bad parking
def outOfBounds(x, y):
    return not ((x <= -4 and y > 3) or (-4 < x < 4 and y > -1) or (x >= 4 and y > 3))

def getYValue(x):
    return 3 if x <= -4 or x >= 4 else -1 if x > -4 and x < 4 else 0

def crossOver(pop1 :individual, pop2 : individual):
    cross = np.random.randint(0, len(pop1))
    second_pivot = np.random.randint(cross, len(pop1))
    child1, child2 = pop2[:cross] + pop1[cross:second_pivot] + pop2[second_pivot:], pop1[:cross] + pop2[cross:second_pivot] + pop1[second_pivot:]
    
    mutate = lambda child: [random.choices([i, (i + 1) % 2], weights=[1 - mutationRate, mutationRate])[0] for i in child]
    return [mutate(child1), mutate(child2)]

# cross breeding
def newPopCrossover(chance_array, pops):
    res = []
    for i in range(math.floor(maxPop/2)):
        new_indexs = np.random.choice(maxPop,2,p=chance_array)
        pop1 = pops[new_indexs[0]]
        pop2 = pops[new_indexs[1]]
        newSeq = crossOver(pop1, pop2)
        res.append(newSeq[0])
        res.append(newSeq[1])
    return res

def getStatus(ind: individual):
    chromesome = np.array_split(ind, total_entries)
    optimization_parameter, acceleration_history, heading_history = [], [], []
    
    for i in range(0, len(chromesome), 2):
        heading, acceleration = getHeading(chromesome[i]), getAcceleration(chromesome[i+1])
        acceleration_history.append(acceleration)
        heading_history.append(heading)
        optimization_parameter.extend([heading, acceleration])

    time = np.linspace(0, time_to_park_car, num=time_to_park_car, endpoint=True)
    _accel = interpolate.CubicSpline(time, acceleration_history, bc_type='natural')(np.linspace(0, 10, 100))
    _heading = interpolate.CubicSpline(time, heading_history, bc_type='natural')(np.linspace(0, 10, 100))
    return _accel, _heading, optimization_parameter

def getChromesomeSequence(ind) -> stateHistory:
    x, y, local_cost = [], [], 0
    currentX, currentY, currentV, currentH = initialState
    acceleration_new, heading_new, optimization_parameter = getStatus(ind)

    for timeStep in range(0, len(all_time_steps_for_x)):
        currentV += acceleration_new[timeStep] * time_step
        currentH += heading_new[timeStep] * time_step
        currentX += currentV * math.cos(currentH) * time_step 
        currentY += currentV * math.sin(currentH) * time_step

        if outOfBounds(currentX, currentY):
            local_cost += math.pow(getYValue(currentX) - currentY, 2 ) * time_step
        x.append(currentX)
        y.append(currentY)

    localFinal = np.array([currentX, currentY, currentH, currentV])
    cost = calcCost(localFinal, local_cost)

    return {
        "cost": cost,
        "fitness": 1 / (cost+1),
        "x": x,
        "y": y,
        "acceleration": acceleration_new,
        "heading": heading_new,
        "optimization_vector": optimization_parameter
    }

def generation(pops, generation):
    fitness, total_combined_fitness, mostFit = [], 0, 0
    for i, pop in enumerate(pops):
        values = getChromesomeSequence(pop)
        total_combined_fitness += values["fitness"]
        fitness.insert(i, values)
        if values["fitness"] > fitness[mostFit]["fitness"]:
            mostFit = i
    print(f"Generation {generation} : J = {fitness[mostFit]['cost']}")

    if fitness[mostFit]["cost"] <= stopCriteria or generation >= maxGenerations:
        return None, fitness[mostFit]

    chance_array = [pop["fitness"] / total_combined_fitness for pop in fitness]
    newGen = newPopCrossover(chance_array, pops)
    newGen.append(pops[mostFit])
    return newGen, fitness[mostFit]

def output(solution):
    print("\nFinal state values:")
    print("x_f = " + str(solution["x"][-1]))
    print("y_f = " + str(solution["y"][-1]))
    print("alpha_f = " + str(solution["heading"][-1]))
    print("v_f = " + str(solution["acceleration"][-1]))

    figure, axis = plt.subplots(5, 1, figsize=(4, 2 * 5), tight_layout= True)

    axis[0].set_xlim([-15, 15])
    axis[0].set_ylim([-10, 15])
    axis[0].plot([-15, -4], [3, 3], 'k-', lw=1)
    axis[0].plot([-4, -4], [-1, 3], 'k-', lw=1)
    axis[0].plot([-4, 4], [-1, -1], 'k-', lw=1)
    axis[0].plot([4, 4], [-1, 3], 'k-', lw=1)
    axis[0].plot([15, 4], [3, 3], 'k-', lw=1)

    axis[0].plot(solution["x"], solution["y"])
    axis[0].plot([-15, -4], [3, 3], 'k-', lw=1)
    axis[0].plot([-4, -4], [-1, 3], 'k-', lw=1)
    axis[0].plot([-4, 4], [-1, -1], 'k-', lw=1)
    axis[0].plot([4, 4], [-1, 3], 'k-', lw=1)
    axis[0].plot([15, 4], [3, 3], 'k-', lw=1)
    plt.setp(axis[0], xlabel="x ", ylabel="y ")

    axis[1].plot(all_time_steps_for_x, solution["acceleration"])
    plt.setp(axis[1], xlabel="Time (s)", ylabel="Acceleration (ft/s^2)")
    axis[2].plot(all_time_steps_for_x, solution["heading"])
    plt.setp(axis[2], xlabel="Time (s)", ylabel="heading (rad/s^2)")
    axis[3].plot(all_time_steps_for_x, solution["x"])
    plt.setp(axis[3], xlabel="Time (s)", ylabel="x (ft)")
    axis[4].plot(all_time_steps_for_x, solution["y"])
    plt.setp(axis[4], xlabel="Time (s)", ylabel="y (ft)")

    plt.show()
    with open('control.dat', 'w') as file:
        file.writelines(f"{val}\n" for val in solution["optimization_vector"])

def main():
    start_time = time.time()
    generations = 0
    currPop = []
    for i in range(0, maxPop):
        currPop.append(newChromosome())
    while (currPop != None and time.time() - start_time < (7 * 60)):
        currPop, res = generation(currPop, generations)
        generations += 1
    output(res)
    
if __name__ == "__main__":
    main()
