from heapq import *
import csv
import networkx as nx
import matplotlib.pyplot as plt
import time
import sys
import argparse


def pars_csv():
    """
    Функция чтения матрицы из файла
    :return:
    """
    data = []

    graph = {}
    with open('graphQ.csv', newline='') as File:
        reader = csv.reader(File)
        for row in reader:
            data.append(row)

    for row in range(1, len(data)):
        count = 0
        let = data[row][0]
        graph_elem = []
        for elem in data[row]:
            if elem == '0':
                count += 1
                continue
            elif elem.isalpha():
                count += 1
                continue
            else:
                index = data[row][elem.index(elem)]
                gr = int(elem), data[0][count]
                count += 1
            graph_elem.append(gr)
        graph[let] = graph_elem

    return graph


def pars_csv_f(path):
    """
    Функция чтения матрицы из csv файла
    :param path: путь до фала
    :return: отформатированный граф
    """
    data = []

    graph = {}
    with open(path, newline='') as File:
        reader = csv.reader(File)
        for row in reader:
            data.append(row)

    for row in range(1, len(data)):
        count = 0
        let = data[row][0]
        graph_elem = []
        for elem in data[row]:
            if elem == '0':
                count += 1
                continue
            elif elem.isalpha():
                count += 1
                continue
            else:
                index = data[row][elem.index(elem)]
                gr = int(elem), data[0][count]
                count += 1
            graph_elem.append(gr)
        graph[let] = graph_elem

    return graph


def dijkstra(start, goal, graph):
    """
    Функция алгоритма Дейкстры
    :param start: начальная вершина
    :param goal: конечная вершина
    :param graph: отформатированные данные графа
    :return:
    """
    queue = []
    heappush(queue, (0, start))
    cost_visited = {start: 0}
    visited = {start: None}
    cost = 0
    while queue:
        cur_cost, cur_node = heappop(queue)
        if cur_node == goal:
            break

        next_nodes = graph[cur_node]
        for next_node in next_nodes:
            neigh_cost, neigh_node = next_node
            new_cost = cost_visited[cur_node] + neigh_cost

            if neigh_node not in cost_visited or new_cost < cost_visited[neigh_node]:
                heappush(queue, (new_cost, neigh_node))
                cost_visited[neigh_node] = new_cost
                visited[neigh_node] = cur_node

    return visited, cost_visited


def plot_graph(data, setting):
    """
    Функция визализации графа
    :param data:
    :param setting:
    :return:
    """
    graph_point = []
    for key in data:
        graph_point.append(key)

    graph_map = set()
    for i in range(len(graph_point)):
        # print(data[graph_point[i]])
        for j in range(len(data[graph_point[i]])):
            tup = (graph_point[i], data[graph_point[i]][j][1], data[graph_point[i]][j][0])
            graph_map.add(tup)

    # создаем экземляр класса
    graph = nx.Graph()

    # добавляем вершины
    for i in range(len(graph_point)):
        graph.add_node(graph_point[i])

    def add_edge(f_item, s_item, graph=None):
        """
        Функция добавления ребер
        :param f_item: Первая вершина
        :param s_item: Вторая вершина
        :param graph: Граф
        :return:
        """
        graph.add_edge(f_item, s_item)
        graph.add_edge(s_item, f_item)

    # создаем ребра
    for i in range(len(graph_point) - 1):
        add_edge(graph_point[i], graph_point[i + 1], graph=graph)

    # добавляем ребра с весами
    graph.add_weighted_edges_from(graph_map)

    nx.draw_circular(graph,
                     node_color='red',
                     node_size=1000,
                     with_labels=True)

    # сохраняем полученный граф в файл
    print(11)
    if setting == 'true':
        print(2)
        plt.savefig('graph.png')
        plt.show()
    elif setting == 'show':
        print(2)
        plt.show()
    elif setting == 'save':
        print(3)
        plt.savefig('graph.png')


def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', default='graphQ.csv')   # путь до файла
    parser.add_argument('-nf', '--nodefrom', default='C')       # вершина начала
    parser.add_argument('-nt', '--nodeto', default='O')         # вершинат конца
    parser.add_argument('-g', '--graph', default='show')        # режим показа графика => show save false true

    return parser


if __name__ == '__main__':
    # -> C; -> O
    start_time = time.time()

    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])
    print(namespace)

    start = '{}'.format(namespace.nodefrom)
    goal = '{}'.format(namespace.nodeto)
    graph = pars_csv_f('{}'.format(namespace.file))
    setting = '{}'.format(namespace.graph)

    if setting != 'false':
        print(1)
        plot_graph(data=graph, setting=setting)

    visited, cost_visited = dijkstra(start, goal, graph)
    cost = 0
    cur_node = goal
    print(f'\npath from {goal} to {start}: \n {goal} ', end='')
    while cur_node != start:
        cur_node = visited[cur_node]
        cost += int(cost_visited[cur_node])
        print(f'---> {cur_node} ', end='')
    print(f"\n {cost}")
    print("\n--- %s seconds general ---" % (time.time() - start_time))
