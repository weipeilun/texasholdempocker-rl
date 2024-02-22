# -*- coding: utf-8-*-

import signal


_interrupted = False


def _signal_handler(signum, frame):
    global _interrupted
    _interrupted = True


def interrupt_callback():
    global _interrupted
    return _interrupted


# capture SIGALRM signal, e.g., signal.alarm()
signal.signal(signal.SIGALRM, _signal_handler)
# capture SIGINT signal, e.g., Ctrl+C
signal.signal(signal.SIGINT, _signal_handler)
# capture SIGTERM signal, e.g., kill
signal.signal(signal.SIGTERM, _signal_handler)