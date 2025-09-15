import threading
from math import floor
from time import sleep

from communication_channel.common import timestamp


class TokenBucket:
    def __init__(self, rate: int, capacity: int):
        """
        :param rate: Token generation rate (tokens per second)
        :param capacity: Maximum number of tokens in the bucket
        """
        self.capacity = capacity  # Maximum bucket size
        self.tokens = capacity  # Start full
        self.rate = rate  # Tokens added per second
        self.last_checked = timestamp()  # Last time tokens were updated
        self.lock = threading.Lock()  # Ensure thread safety

    def consume(self, amount):
        """
        Attempt to consume 'amount' tokens. Return True if successful, False otherwise.
        """
        with self.lock:
            self._add_tokens()  # Refill tokens before checking availability
            if self.tokens >= amount:
                self.tokens -= amount
                return True
            return False  # Not enough tokens

    def wait_for_tokens(self, amount):
        """
        Block until 'amount' tokens are available.
        """
        while True:
            if self.consume(amount):
                return
            sleep(0.1)

    def _add_tokens(self):
        """
        Add tokens based on elapsed time since last update.
        """
        now = timestamp()
        elapsed = now - self.last_checked
        self.last_checked = now
        new_tokens = floor(elapsed * self.rate)  # Tokens generated in elapsed time
        self.tokens = min(self.capacity, self.tokens + new_tokens)  # Cap at bucket size

