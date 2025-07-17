import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import imageio
import os
import pandas as pd
from math import sqrt
from itertools import groupby, product
from src.utils import parse_walls
from src.plots.FlowGraph import FlowGraph
from src.plots.GroupedLanesGraph import GroupedLanesGraph
from src.plots.DensityHeatmap import DensityHeatmap
from src.plots.DensityRowGraph import DensityRowGraph
from src.plots.FlowGraphInfections import FlowGraphInfections


def benchmark_graph():
    results = {
        'N': [100, 200, 400, 800, 1600],
        'mmoc': [7.09, 17.46, 48.89, 156.24, 551.06],
        'retqss': [2.77, 5.06, 10.98, 25.70, 67.70]
    }

    description = """
    The simulations were made using the same parameters for both models. 
    Same ones used by Helbing and Molnar 1995. No obstacles were used.
    The number of volumes for the retqss implementation was 100.
    """

    plt.figure(figsize=(10, 10))
    plt.plot(results['N'], results['mmoc'], label='mmoc only')
    plt.plot(results['N'], results['retqss'], label='mmoc+retqss')
    plt.title('Comparacion de tiempos de ejecucion*')
    plt.xlabel('Numero de peatones en la simulacion')
    plt.ylabel('Tiempo(s)')
    plt.legend()
    plt.savefig(f'benchmark_graph.png')
    plt.close()

class Plotter:
    """
    This class contains the functions to plot the simulation results.
    """
    def flow_graph(self, solution_file, output_dir, parameters):
        flow_graph = FlowGraph(solution_file, output_dir, parameters)
        flow_graph.plot()

    def grouped_lanes_graph(self, results, output_dir):
        grouped_lanes_graph = GroupedLanesGraph(results, output_dir)
        grouped_lanes_graph.plot()

    def density_heatmap(self, solution_file, output_dir):
        density_heatmap = DensityHeatmap(solution_file, output_dir)
        density_heatmap.plot()

    def density_row_graph(self, solution_file, output_dir):
        density_row_graph = DensityRowGraph(solution_file, output_dir)
        density_row_graph.plot()
    
    def flow_graph_infections(self, solution_file, output_dir, parameters):
        flow_graph_infections = FlowGraphInfections(solution_file, output_dir, parameters)
        flow_graph_infections.plot()