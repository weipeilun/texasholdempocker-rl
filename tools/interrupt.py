# -*- coding: utf-8-*-

import signal


_interrupted = False


def _signal_handler(signum, frame):
    global _interrupted
    _interrupted = True


def interrupt_callback():
    global _interrupted
    return _interrupted


# capture SIGINT signal, e.g., Ctrl+C
signal.signal(signal.SIGALRM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)