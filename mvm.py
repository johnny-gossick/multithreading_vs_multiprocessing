from threading import Thread
from random import shuffle
from multiprocessing import Process, Value
from datetime import datetime
import numpy as np
import pandas as pd


class AddingProcess(Process):
    def __init__(self, numbers_to_add):
        Process.__init__(self)
        self.numbers_to_add = numbers_to_add
        self.sum = Value('d', 0.0)

    def run(self) -> None: self.sum.value = sum(self.numbers_to_add)

    def total(self): return self.sum.value


class AddingThread(Thread):
    def __init__(self, numbers_to_add):
        Thread.__init__(self)
        self.numbers_to_add = numbers_to_add
        self.sum = 0.0

    def run(self) -> None: self.sum = sum(self.numbers_to_add)

    def total(self): return self.sum


def get_elapsed_time(start): return (datetime.now() - start).total_seconds()


def get_random_numbers(number_count):
    return np.array(shuffle(list(range(number_count))))


def time_adding_serially(numbers_to_add):
    start = datetime.now()
    total = sum(numbers_to_add)

    return get_elapsed_time(start), total


def time_adding_numpy(numbers_to_add):
    start = datetime.now()
    total = np.sum(numbers_to_add)

    return get_elapsed_time(start), total


def time_adding_concurrently(numbers_to_add, task_count, task_class_name):
    start = datetime.now()
    numbers_to_add_lists = np.array_split(numbers_to_add, task_count)
    tasks = [task_class_name(numbers) for numbers in numbers_to_add_lists]
    [task.start() for task in tasks]
    [task.join() for task in tasks]
    total = sum([task.total() for task in tasks])

    return get_elapsed_time(start), total


def run_experiment(number_count, task_count):
    numbers_to_add = get_random_numbers(number_count)
    serial_time, serial_total = time_adding_serially(numbers_to_add)
    multiprocess_time, multiprocess_total = time_adding_concurrently(numbers_to_add, task_count, AddingProcess)
    multithreading_time, multithreading_total = time_adding_concurrently(numbers_to_add, task_count, AddingThread)
    numpy_time, numpy_total = time_adding_numpy(numbers_to_add)

    return pd.DataFrame({
        'mode': ['serial', 'multiprocessing', 'multithreading', 'numpy'],
        'time': [serial_time, multiprocess_time, multithreading_time, numpy_time],
        'sum':  [serial_total, multiprocess_total, multithreading_total, numpy_total],
        'data_size': [number_count, number_count, number_count, number_count],
        'task_count': [1, task_count, task_count, None]
    })


def main():
    df = run_experiment(1000000000, 8)
    df.to_csv('results.csv')
    print(df)


if __name__ == '__main__':
    main()
