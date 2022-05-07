from tracemalloc import start
from airquality import AirQuality, AirQualityGraph


def main():
    aq = AirQuality()
    graph = aq.create_subgraph(start_node = 75, size = 10)
    graph.print_info()

if __name__ == '__main__':
    main()
