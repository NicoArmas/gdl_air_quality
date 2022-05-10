from tracemalloc import start
from airquality import AirQuality


def main():
    aq = AirQuality()
    print(aq.lookup_id(3))
    print(aq.lookup_index('1.0-5.0-1.0'))


if __name__ == '__main__':
    main()
