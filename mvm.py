from threading import Thread
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


def get_random_numbers(number_count): return np.array(range(number_count))


def get_speedup(original, other): return float(original) / float(other)


def time_adding_serially(numbers_to_add):
    start = datetime.now()
    total = sum(numbers_to_add)

    return get_elapsed_time(start), total


def time_adding_numpy(numbers_to_add):
    start = datetime.now()
    total = np.sum(numbers_to_add)

    return get_elapsed_time(start), total


def time_adding_pandas(numbers_to_add):
    df = pd.DataFrame({'numbers': numbers_to_add})
    start = datetime.now()
    total = df.sum()

    return get_elapsed_time(start), total['numbers']


def time_adding_concurrently(numbers_to_add, task_count, task_class_name):
    start = datetime.now()
    numbers_to_add_lists = np.array_split(numbers_to_add, task_count)
    tasks = [task_class_name(numbers) for numbers in numbers_to_add_lists]
    [task.start() for task in tasks]
    [task.join() for task in tasks]
    total = sum([task.total() for task in tasks])

    return get_elapsed_time(start), total


def run_experiment(number_count, task_count):
    print('Generating numbers...')
    numbers_to_add = get_random_numbers(number_count)
    print('Adding serially...')
    serial_time, serial_total = time_adding_serially(numbers_to_add)
    print('Adding w/ multiprocessing...')
    multiprocess_time, multiprocess_total = time_adding_concurrently(numbers_to_add, task_count, AddingProcess)
    print('Adding w/ multithreading...')
    multithreading_time, multithreading_total = time_adding_concurrently(numbers_to_add, task_count, AddingThread)
    print('Adding w/ numpy...')
    numpy_time, numpy_total = time_adding_numpy(numbers_to_add)
    print('Adding w/ pandas...')
    pandas_time, pandas_total = time_adding_pandas(numbers_to_add)

    time_list = [serial_time, multiprocess_time, multithreading_time, numpy_time, pandas_time]

    return pd.DataFrame({
        'mode': ['serial', 'multiprocessing', 'multithreading', 'numpy', 'pandas'],
        'speedup': [get_speedup(serial_time, other_time) for other_time in time_list],
        'time': time_list,
        'parallel_factor': [1, task_count, task_count, None, None],
        'sum':  [serial_total, multiprocess_total, multithreading_total, numpy_total, pandas_total],
        'data_size': [number_count, number_count, number_count, number_count, number_count]
    }).sort_values(by=['time', 'mode'], ascending=True)


def main():
    df = run_experiment(100000000, 4)
    df.to_csv('results.csv')
    print(df)


if __name__ == '__main__':
    main()
