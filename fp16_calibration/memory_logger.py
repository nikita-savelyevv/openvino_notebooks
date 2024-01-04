import atexit
import queue
import threading
import time
import datetime
from pathlib import Path

import matplotlib.pyplot as plt

import psutil


class MemoryLogger:
    def __init__(self, log_dir: Path, plot_title: str = ""):
        self.log_dir = log_dir
        self.plot_title = plot_title
        self.monitoring_thread_should_stop = False
        self.monitoring_in_progress = False

        self.memory_monitor_thread = None
        self.memory_data_queue = None

    def start_logging(self):
        if self.monitoring_in_progress:
            raise Exception("Monitoring already in progress")

        self.memory_data_queue = queue.Queue()
        self.monitoring_thread_should_stop = False

        self.memory_monitor_thread = threading.Thread(target=self._monitor_memory)
        self.memory_monitor_thread.daemon = True  # Daemonize the thread
        self.memory_monitor_thread.start()
        atexit.register(lambda: self.stop_logging())

        self.monitoring_in_progress = True

        return self

    def stop_logging(self):
        self.monitoring_thread_should_stop = True
        self.monitoring_in_progress = False
        self.memory_monitor_thread.join()
        self.log_memory_usage()

    def log_memory_usage(self):
        memory_usage_data = list(self.memory_data_queue.queue)

        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True)

        # Save the memory usage data to a file
        with open(self.log_dir / 'memory_usage_log.txt', 'w') as log_file:
            for timestamp, memory_usage in memory_usage_data:
                log_file.write(f"{timestamp} {memory_usage}\n")

            log_file.writelines([
                f"Total time: {memory_usage_data[-1][0] - memory_usage_data[0][0]}\n",
                f"Max memory: {max(tuple(zip(*memory_usage_data))[1])} (MB)"])

        timestamps, memory_usage = zip(*memory_usage_data)
        fig = plt.figure(figsize=(10, 6))
        plt.plot(timestamps, memory_usage)
        plt.xlabel("Time")
        plt.ylabel("Memory Usage (MB)")
        plt.title(f"Memory Usage vs. Time ({self.plot_title})")
        plt.grid(True)
        plt.savefig(self.log_dir / "memory_usage.png")
        plt.close(fig)

    def _monitor_memory(self):
        while not self.monitoring_thread_should_stop:
            memory_usage = psutil.Process().memory_info().rss >> 20     # MB
            self.memory_data_queue.put((datetime.datetime.now(), memory_usage))
            time.sleep(1)


if __name__ == '__main__':
    memory_logger = MemoryLogger(Path("./logs")).start_logging()
    import numpy as np
    from tqdm import tqdm
    a = []
    for i in tqdm(range(1000)):
        a.append(np.random.random((1000000)))
        memory_logger.log_memory_usage()
        time.sleep(0.1)
    