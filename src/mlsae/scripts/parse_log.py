#!/usr/bin/env python3

import sys
import re
from datetime import datetime
from typing import Dict, Tuple

def timestamp_to_seconds(timestamp: str) -> int:
    """Convert timestamp to seconds since midnight."""
    time_str = timestamp.split()[1].split('.')[0]  # Remove milliseconds
    time_obj = datetime.strptime(time_str, '%H:%M:%S')
    return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second

def parse_log(log_file: str) -> None:
    # Initialize variables for each worker
    start_times: Dict[int, int] = {}
    wait_start_times: Dict[int, int] = {}
    total_wait_times: Dict[int, int] = {i: 0 for i in range(8)}  # Support 8 workers
    is_waiting: Dict[int, bool] = {i: False for i in range(8)}   # Support 8 workers
    last_timestamp = 0

    # Compile regex patterns
    log_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\.\d{3}).*\[INFO\]\s(.*)$')
    worker_start_pattern = re.compile(r'^Starting worker on device cuda:([0-7])$')  # Support cuda:0 through cuda:7
    worker_waiting_pattern = re.compile(r'^GPU worker ([0-7]) waiting for task$')   # Support workers 0-7
    worker_got_task_pattern = re.compile(r'^GPU worker ([0-7]) got task$')         # Support workers 0-7

    try:
        with open(log_file, 'r') as f:
            for line in f:
                # Extract timestamp and message
                match = log_pattern.match(line)
                if not match:
                    continue

                timestamp, message = match.groups()
                seconds = timestamp_to_seconds(timestamp)
                last_timestamp = seconds

                # Track worker start times
                if (match := worker_start_pattern.match(message)):
                    worker = int(match.group(1))
                    start_times[worker] = seconds
                    print(f"Worker {worker} started at {timestamp}")

                # Track waiting periods
                elif (match := worker_waiting_pattern.match(message)):
                    worker = int(match.group(1))
                    if not is_waiting[worker]:
                        wait_start_times[worker] = seconds
                        is_waiting[worker] = True

                elif (match := worker_got_task_pattern.match(message)):
                    worker = int(match.group(1))
                    if is_waiting[worker]:
                        wait_duration = seconds - wait_start_times[worker]
                        total_wait_times[worker] += wait_duration
                        is_waiting[worker] = False

        # Calculate and print results
        for worker in range(8):  # Check all 8 possible workers
            if worker in start_times:
                alive_time = last_timestamp - start_times[worker]
                wait_time = total_wait_times[worker]
                wait_percentage = (wait_time / alive_time) * 100

                print(f"GPU Worker {worker}:")
                print(f"  Total alive time: {alive_time} seconds")
                print(f"  Total wait time: {wait_time} seconds")
                print(f"  Wait percentage: {wait_percentage:.2f}%")

    except FileNotFoundError:
        print(f"Error: Log file '{log_file}' not found")
        sys.exit(1)

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path_to_log_file>")
        sys.exit(1)

    parse_log(sys.argv[1])

if __name__ == "__main__":
    main()
