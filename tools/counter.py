from threading import Lock


class Counter:
    def __init__(self):
        self.value = 0
        self.lock = Lock()

    def increment(self):
        with self.lock:
            self.value += 1
            return self.value

    def decrement(self):
        with self.lock:
            self.value -= 1
            return self.value

    def reset(self):
        with self.lock:
            self.value = 0

    def get_value(self):
        with self.lock:
            return self.value
