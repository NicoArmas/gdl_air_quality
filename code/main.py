from operator import is_
from tracemalloc import start
from airquality import AirQuality


def main():
    aq = AirQuality(is_subgraph=True, sub_start='6.0-73.0-1201.0', sub_size=100)
    print(aq)


if __name__ == '__main__':
    main()
