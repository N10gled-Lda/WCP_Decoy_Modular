import queue
import threading
from queue import Queue


class ByteQueue:
    """
    A thread-safe queue where the capacity is defined in bytes rather than the number of elements.
    """

    def __init__(self, capacity_bytes: int):
        """
        Initializes the ByteQueue.

        :param capacity_bytes: The maximum capacity of the queue, measured in bytes.
        """
        self._queue = Queue()
        self.capacity = capacity_bytes
        self.current_capacity = self.capacity
        self._current_capacity_lock = threading.Lock()
        self._current_capacity_change = threading.Condition(self._current_capacity_lock)


    def put(self, data: bytes, block: bool = True, timeout: float | None = None) -> None:
        """
        Inserts data into the queue, respecting byte capacity constraints.

        :param data: The data (bytes) to be added to the queue.
                     If the data size exceeds 'capacity', a ValueError is raised.
        :param block: If True (default), waits until enough space is available.
                      If False, attempts immediate insertion and raises queue.Full if there's insufficient space.
                      If True and 'timeout' is specified, waits up to 'timeout' seconds for space,
                      then raises queue.Full if space isn't available.
        :param timeout: Maximum wait time (in seconds) if 'block' is True. Ignored if 'block' is False.
        :raises ValueError: If the data size exceeds the queue's maximum capacity.
        :raises queue.Full: If the queue lacks space for the data and 'block' is False or timeout expires.
        """
        data_size = len(data)
        if data_size > self.capacity:
            raise ValueError(f"Data Size ({data_size} bytes) is larger than the maximum queue capacity ({self.capacity} bytes).")

        with self._current_capacity_change:
            if not data_size <= self.current_capacity:
                if not block:
                    raise queue.Full
                else:
                    has_changed = self._current_capacity_change.wait(timeout)
                    if not has_changed:
                            raise TimeoutError
            self._queue.put(data, block=False)
            self.current_capacity -= data_size
            self._current_capacity_change.notify_all()

    def get(self, block: bool = True, timeout: float | None = None) -> bytes | None:
        """
        Retrieves and removes an item from the queue.

        :param block: If True (default), waits until data is available.
                      If False, tries to get data immediately and raises queue.Empty if the queue is empty.
                      If True and 'timeout' is specified, waits up to 'timeout' seconds for data,
                      then raises queue.Empty if data isn't available.
        :param timeout: Maximum wait time (in seconds) if 'block' is True. Ignored if 'block' is False.
        :raises queue.Empty: If the queue is empty and 'block' is False or if 'block' is True and 'timeout' expires.
        :return: The retrieved data (bytes) from the queue.
        """
        data = self._queue.get(block, timeout)
        with self._current_capacity_change:
            self.current_capacity += len(data)
            self._current_capacity_change.notify_all()
            return data
