import time
from multiprocessing import shared_memory, Manager
from abc import abstractmethod
import numpy as np
import math
import logging
from threading import Thread
from queue import Empty, Queue
from tools import interrupt

# high performance process-secure queue for arrays using shared_memory
class AbstractQueue(object):

    def __int__(self, element_size):
        super(AbstractQueue).__init__()

    @abstractmethod
    def put(self, data):
        pass

    @abstractmethod
    def get(self, block=True, timeout=None):
        pass

    @abstractmethod
    def qsize(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def unlink(self):
        pass

    def _put_trigger_queue(self, trigger, trigger_queue, block):
        if block:
            trigger_queue.put(trigger)
        else:
            try:
                trigger_queue.get(block=False)
            except Empty:
                pass
            trigger_queue.put(trigger)


class One2OneQueue(AbstractQueue):

    def __init__(self, element_shape, element_dtype, shm=None, trigger_queue=None, signal_queue=None, signal_to_send=None, max_queue_size=100, ignore_array=False):
        assert isinstance(element_shape, (tuple, list)), 'element_shape must be a list or tuple'

        if not isinstance(element_shape[0], (tuple, list)):
            element_shape = [element_shape]
        if not isinstance(element_dtype, (tuple, list)):
            element_dtype = [element_dtype]
        assert len(element_shape) == len(element_dtype), 'Length of element_shape must equals to length of element_dtype'

        for e_shape, e_dtype in zip(element_shape, element_dtype):
            assert isinstance(e_shape, (tuple, list)), 'element_shape must be a list or tuple'
            assert np.issubdtype(e_dtype, np.generic), 'element_dtype must be a numpy dtype'

        if not ignore_array:
            if shm is not None:
                if not isinstance(shm, (tuple, list)):
                    shm = [shm]
                self.shm = shm
            else:
                self.shm = []
                for e_shape, e_dtype in zip(element_shape, element_dtype):
                    element_size = math.prod(e_shape) * e_dtype.itemsize
                    self.shm.append(shared_memory.SharedMemory(create=True, size=element_size))

            assert len(self.shm) == len(element_shape), 'Length of shm must equals to length of element_shape'
            self.shm_np = []
            for e_shape, e_dtype, shm in zip(element_shape, element_dtype, self.shm):
                self.shm_np.append(np.ndarray(shape=e_shape, dtype=e_dtype, buffer=shm.buf))

            if signal_queue is None:
                signal_queue = Manager().Queue(maxsize=max_queue_size)

        if trigger_queue is None:
            trigger_queue = Manager().Queue(maxsize=max_queue_size)

        self.trigger_queue = trigger_queue
        self.signal_queue = signal_queue
        self.signal_to_send = signal_to_send
        self.ignore_array = ignore_array

        self.signal_queue.put(self.signal_to_send)

    def put(self, data, block=True):
        if self.ignore_array:
            trigger = data
        else:
            assert isinstance(data, (list, tuple)) and len(data) == 2, 'Data should contain and only contain trigger and array'
            assert isinstance(data[1], np.ndarray) or (isinstance(data[1], (list, tuple)) and len(data[1]) == len(self.shm_np) and all([isinstance(item, np.ndarray) for item in data[1]])), 'Only ndarray data is supported.'
            trigger, arrs = data

            if not isinstance(arrs, (list, tuple)):
                arrs = [arrs]

            if block:
                signal = self.signal_queue.get(block=True)
            else:
                signal = None
                try:
                    signal = self.signal_queue.get(block=False)
                except Empty:
                    pass

            for arr, shm_np in zip(arrs, self.shm_np):
                np.copyto(shm_np, arr)

        self._put_trigger_queue(trigger, self.trigger_queue, block)

        if not self.ignore_array:
            return signal
        else:
            return

    def get(self, block=True, timeout=None, ignore_array=False):
        trigger = self.trigger_queue.get(block=block, timeout=timeout)

        if self.ignore_array:
            # array取值全部忽略
            return trigger
        elif ignore_array:
            # 本次array取值忽略
            self.signal_queue.put(self.signal_to_send)

            return trigger
        else:
            # 不忽略array取值
            if len(self.shm_np) == 1:
                datas = np.copy(self.shm_np[0])
            else:
                datas = []
                for shm_np in self.shm_np:
                    datas.append(np.copy(shm_np))

            self.signal_queue.put(self.signal_to_send)

            return trigger, datas

    def qsize(self):
        return int(not self.trigger_queue.empty())

    def close(self):
        if not self.ignore_array:
            for shm in self.shm:
                shm.close()

    def unlink(self):
        if not self.ignore_array:
            for shm in self.shm:
                shm.unlink()


class One2ManyQueue(AbstractQueue):

    class One2OneQueueOverProcess:
        class One2OneQueue(One2OneQueue):

            def __init__(self, element_shape, element_dtype, queue_idx_over_process, queue_idx_in_process, shm=None, trigger_queue=None, signal_queue=None, signal_to_send=None, max_queue_size=100, ignore_array=False):
                super().__init__(element_shape=element_shape,
                                 element_dtype=element_dtype,
                                 shm=shm,
                                 trigger_queue=trigger_queue,
                                 signal_queue=signal_queue,
                                 signal_to_send=signal_to_send,
                                 max_queue_size=max_queue_size,
                                 ignore_array=ignore_array,
                                 )

                self.queue_idx_over_process = queue_idx_over_process
                self.queue_idx_in_process = queue_idx_in_process

            def get(self, block=True, timeout=None, ignore_array=False):
                tid, trigger = self.trigger_queue.get(block=block, timeout=timeout)
                assert tid == self.signal_to_send, ValueError(f'One2ManyQueue.One2OneQueueOverProcess.One2OneQueue.signal_to_send={self.signal_to_send} but tid={tid}')

                if self.ignore_array:
                    # array取值全部忽略
                    return tid, trigger
                elif ignore_array:
                    # 本次array取值忽略
                    self.signal_queue.put(self.signal_to_send)

                    return tid, trigger
                else:
                    # 不忽略array取值
                    if len(self.shm_np) == 1:
                        datas = np.copy(self.shm_np[0])
                    else:
                        datas = []
                        for shm_np in self.shm_np:
                            datas.append(np.copy(shm_np))

                    self.signal_queue.put(self.signal_to_send)

                    return tid, trigger, datas

        def __init__(self, element_shape, element_dtype, queue_idx_over_process, n_consumers_in_process, shms, signal_queue, tigger_queue_over_process, max_queue_size=100, ignore_array=False):
            self.queue_idx_over_process = queue_idx_over_process
            self.n_consumers_in_process = n_consumers_in_process
            self.trigger_queue_over_process = tigger_queue_over_process

            self.trigger_queue_in_process_list = [Queue() for _ in range(n_consumers_in_process)]

            self.consumer_list = [self.One2OneQueue(element_shape=element_shape, element_dtype=element_dtype, queue_idx_over_process=queue_idx_over_process, queue_idx_in_process=idx, shm=shm, trigger_queue=trigger_queue_in_process, signal_queue=signal_queue, signal_to_send=queue_idx_over_process * n_consumers_in_process + idx, max_queue_size=max_queue_size, ignore_array=ignore_array) for idx, (trigger_queue_in_process, shm) in enumerate(zip(self.trigger_queue_in_process_list, shms))]

            self.thread_started = False

            self.thread_id_map = {i: i % n_consumers_in_process for i in range(queue_idx_over_process * n_consumers_in_process, (queue_idx_over_process + 1) * n_consumers_in_process)}

        def start_map_data_thread(self):
            if not self.thread_started:
                Thread(target=self.map_data_thread, daemon=True).start()
                self.thread_started = True

        def map_data_thread(self):
            while True:
                if interrupt.interrupt_callback():
                    logging.info(f"One2ManyQueue_{self.queue_idx_over_process}.map_data_thread detect interrupt")
                    break

                try:
                    tid, trigger = self.trigger_queue_over_process.get(block=True, timeout=0.001)
                    self.trigger_queue_in_process_list[self.thread_id_map[tid]].put((tid, trigger))
                except Empty:
                    pass

    def __init__(self, element_shape, element_dtype, n_consumers_over_process, n_consumers_in_process, max_queue_size=100, ignore_array=False, verbose=0):
        assert isinstance(element_shape, (tuple, list)), 'element_shape must be a list or tuple'

        if not isinstance(element_shape[0], (tuple, list)):
            element_shape = [element_shape]
        if not isinstance(element_dtype, (tuple, list)):
            element_dtype = [element_dtype]
        assert len(element_shape) == len(element_dtype), 'Length of element_shape must equals to length of element_dtype'

        for e_shape, e_dtype in zip(element_shape, element_dtype):
            assert isinstance(e_shape, (tuple, list)), 'element_shape must be a list or tuple'
            assert np.issubdtype(e_dtype, np.generic), 'element_dtype must be a numpy dtype'
        assert isinstance(n_consumers_over_process, int), 'n_consumers_over_process must be a int'
        assert isinstance(n_consumers_in_process, int), 'n_consumers_in_process must be a int'

        self.n_consumers_over_process = n_consumers_over_process
        self.n_consumers_in_process = n_consumers_in_process
        self.ignore_array = ignore_array
        self.verbose = verbose

        self.signal_queue = Manager().Queue(maxsize=n_consumers_over_process * n_consumers_in_process)
        self.trigger_queue_list = [Manager().Queue(maxsize=n_consumers_over_process * n_consumers_in_process) for _ in range(self.n_consumers_over_process)]

        self.shms_list = list()
        self.shms_np_list = list()
        for _ in range(self.n_consumers_over_process):
            shms = list()
            shms_np = list()
            for _ in range(self.n_consumers_in_process):
                shared_memory_list = []
                shm_np_list = []
                for e_shape, e_dtype in zip(element_shape, element_dtype):
                    element_size = math.prod(e_shape) * e_dtype.itemsize
                    shm = shared_memory.SharedMemory(create=True, size=element_size)
                    shared_memory_list.append(shm)
                    shm_np_list.append(np.ndarray(shape=e_shape, dtype=e_dtype, buffer=shm.buf))

                shms.append(shared_memory_list)
                shms_np.append(shm_np_list)

            self.shms_list.append(shms)
            self.shms_np_list.append(shms_np)

        self.consumer_list = [self.One2OneQueueOverProcess(element_shape=element_shape, element_dtype=element_dtype, queue_idx_over_process=idx, n_consumers_in_process=self.n_consumers_in_process, shms=shms, signal_queue=self.signal_queue, tigger_queue_over_process=trigger_queue, max_queue_size=max_queue_size, ignore_array=self.ignore_array) for idx, (trigger_queue, shms) in enumerate(zip(self.trigger_queue_list, self.shms_list))]

        self.signal_slots = np.ones([n_consumers_over_process * n_consumers_in_process, ], dtype=np.bool_)
        self.thread_id_map = {i: (i // n_consumers_in_process, i % n_consumers_in_process)for i in range(n_consumers_over_process * n_consumers_in_process)}

        self.thread_started = False

    def start_receive_signal_thread(self):
        if not self.thread_started:
            Thread(target=self.receive_signal_thread, daemon=True).start()
            self.thread_started = True

    def receive_signal_thread(self):
        while True:
            if interrupt.interrupt_callback():
                logging.info(f"One2ManyQueue.receive_signal_thread detect interrupt")
                break

            try:
                tid = self.signal_queue.get(block=True, timeout=0.001)
                self.signal_slots[tid] = True
            except Empty:
                pass

    def put(self, data):
        assert isinstance(data, (list, tuple)) and len(data) == 3, 'Data should contain and only contain target_queue_idx, trigger and array'
        assert isinstance(data[2], np.ndarray) or (isinstance(data[2], (list, tuple)) and len(data[2]) == len(self.shms_np_list[0]) and all([isinstance(item, np.ndarray) for item in data[2]])), 'Only ndarray data is supported.'
        tid, trigger, arrs = data

        if not isinstance(arrs, (list, tuple)):
            arrs = [arrs]

        while not self.signal_slots[tid]:
            if interrupt.interrupt_callback():
                logging.info(f"One2ManyQueue.map_data_thread detect interrupt")
                raise InterruptedError

            time.sleep(0.001)

        over_process_idx, in_process_idx = self.thread_id_map[tid]
        shms_np = self.shms_np_list[over_process_idx][in_process_idx]
        for arr, shm_np in zip(arrs, shms_np):
            np.copyto(shm_np, arr)

        self.signal_slots[tid] = False

        self.trigger_queue_list[over_process_idx].put((tid, trigger))

    def get(self, block=True, timeout=None):
        raise NotImplementedError('One2ManyQueue.consumer_list.get() should be called instead of One2ManyQueue.get()')

    def qsize(self):
        raise NotImplementedError('One2ManyQueue.consumer_list.qsize() should be called instead of One2ManyQueue.qsize()')

    def close(self):
        for shm in self.shm_list:
            shm.close()
        self.signal_queue.close()
        for trigger_queue in self.trigger_queue_list:
            trigger_queue.close()

    def unlink(self):
        for shm in self.shm_list:
            shm.unlink()


class Many2OneQueue(AbstractQueue):

    class One2OneQueueOverProcess:
        class One2OneQueue(One2OneQueue):

            def __init__(self, element_shape, element_dtype, queue_idx_over_process, queue_idx_in_process, shm=None, trigger_queue=None, signal_queue=None, signal_to_send=None, max_queue_size=100, ignore_array=False):
                super().__init__(element_shape=element_shape,
                                 element_dtype=element_dtype,
                                 shm=shm,
                                 trigger_queue=trigger_queue,
                                 signal_queue=signal_queue,
                                 signal_to_send=signal_to_send,
                                 max_queue_size=max_queue_size,
                                 ignore_array=ignore_array,
                                 )

                self.queue_idx_over_process = queue_idx_over_process
                self.queue_idx_in_process = queue_idx_in_process

            def put(self, data, block=True):
                if self.ignore_array:
                    super().put((self.queue_idx_over_process, self.queue_idx_in_process, self.signal_to_send, data))
                else:
                    assert isinstance(data, (list, tuple)) and len(data) == 2, 'Data should contain and only contain trigger and array'
                    assert isinstance(data[1], np.ndarray) or (isinstance(data[1], (list, tuple)) and len(data[1]) == len(self.shm_np) and all([isinstance(item, np.ndarray) for item in data[1]])), 'Only ndarray data is supported.'
                    trigger, arrs = data
                    super().put(((self.queue_idx_over_process, self.queue_idx_in_process, self.signal_to_send, trigger), arrs))

        def __init__(self, element_shape, element_dtype, queue_idx_over_process, n_producers_in_process, shms, trigger_queue, signal_queue_over_process, max_queue_size=100, ignore_array=False):
            self.queue_idx_over_process = queue_idx_over_process
            self.n_producers_in_process = n_producers_in_process
            self.signal_queue_over_process = signal_queue_over_process

            self.signal_queue_in_process_list = [Queue() for _ in range(n_producers_in_process)]

            self.producer_list = [self.One2OneQueue(element_shape=element_shape, element_dtype=element_dtype, queue_idx_over_process=queue_idx_over_process, queue_idx_in_process=idx, shm=shm, trigger_queue=trigger_queue, signal_queue=signal_queue_in_process, signal_to_send=queue_idx_over_process * n_producers_in_process + idx, max_queue_size=max_queue_size, ignore_array=ignore_array) for idx, (signal_queue_in_process, shm) in enumerate(zip(self.signal_queue_in_process_list, shms))]

            self.thread_started = False

            for idx, producer in enumerate(self.producer_list):
                producer.signal_queue.put(producer.signal_to_send)

        def start_map_data_thread(self):
            if not self.thread_started:
                Thread(target=self.map_data_thread, daemon=True).start()
                self.thread_started = True

        def map_data_thread(self):
            while True:
                if interrupt.interrupt_callback():
                    logging.info(f"Many2OneQueue_{self.queue_idx_over_process}.map_data_thread detect interrupt")
                    break

                try:
                    in_process_tid, tid = self.signal_queue_over_process.get(block=True, timeout=0.001)
                    in_process_producer = self.producer_list[in_process_tid]
                    assert in_process_producer.signal_to_send == tid, ValueError(f'in_process_producer.signal_to_send={in_process_producer.signal_to_send} but tid={tid}')
                    in_process_producer.signal_queue.put(tid)
                except Empty:
                    pass

    def __init__(self, element_shape, element_dtype, n_producers_over_process, n_producers_in_process, max_queue_size=100, verbose=0):
        assert isinstance(element_shape, (tuple, list)), 'element_shape must be a list or tuple'

        if not isinstance(element_shape[0], (tuple, list)):
            element_shape = [element_shape]
        if not isinstance(element_dtype, (tuple, list)):
            element_dtype = [element_dtype]
        assert len(element_shape) == len(element_dtype), 'Length of element_shape must equals to length of element_dtype'

        for e_shape, e_dtype in zip(element_shape, element_dtype):
            assert isinstance(e_shape, (tuple, list)), 'element_shape must be a list or tuple'
            assert np.issubdtype(e_dtype, np.generic), 'element_dtype must be a numpy dtype'
        assert isinstance(n_producers_over_process, int), 'n_producers_over_process must be a int'
        assert isinstance(n_producers_in_process, int), 'n_consumers_in_process must be a int'

        self.n_producers_over_process = n_producers_over_process
        self.n_producers_in_process = n_producers_in_process
        self.verbose = verbose

        self.signal_queue_list = [Manager().Queue(maxsize=n_producers_over_process * n_producers_in_process) for _ in range(self.n_producers_over_process)]
        self.trigger_queue = Manager().Queue(maxsize=n_producers_over_process * n_producers_in_process)

        self.shms_list = list()
        self.shms_np_list = list()
        for _ in range(self.n_producers_over_process):
            shms = list()
            shms_np = list()
            for _ in range(self.n_producers_in_process):
                shared_memory_list = []
                shm_np_list = []
                for e_shape, e_dtype in zip(element_shape, element_dtype):
                    element_size = math.prod(e_shape) * e_dtype.itemsize
                    shm = shared_memory.SharedMemory(create=True, size=element_size)
                    shared_memory_list.append(shm)
                    shm_np_list.append(np.ndarray(shape=e_shape, dtype=e_dtype, buffer=shm.buf))

                shms.append(shared_memory_list)
                shms_np.append(shm_np_list)

            self.shms_list.append(shms)
            self.shms_np_list.append(shms_np)

        self.producer_list = [self.One2OneQueueOverProcess(element_shape=element_shape, element_dtype=element_dtype, queue_idx_over_process=idx, n_producers_in_process=self.n_producers_in_process, shms=shms, trigger_queue=self.trigger_queue, signal_queue_over_process=signal_queue_over_process, max_queue_size=max_queue_size) for idx, (signal_queue_over_process, shms) in enumerate(zip(self.signal_queue_list, self.shms_list))]

    def put(self, data, broadcast_to=None, block=True):
        raise NotImplementedError('Many2OneQueue.producer_list.put() should be called instead of Many2OneQueue.put()')

    def get(self, block=True, timeout=None):
        over_process_idx, in_process_idx, signal_to_send, trigger = self.trigger_queue.get(block=block, timeout=timeout)

        shms_np = self.shms_np_list[over_process_idx][in_process_idx]

        if len(shms_np) == 1:
            datas = np.copy(shms_np[0])
        else:
            datas = []
            for shm_np in shms_np:
                datas.append(np.copy(shm_np))

        self.signal_queue_list[over_process_idx].put((in_process_idx, signal_to_send))

        return trigger, datas

    def qsize(self):
        return self.trigger_queue.qsize()

    def close(self):
        for shms in self.shms_list:
            for shm in shms:
                shm.close()

    def unlink(self):
        for shms in self.shms_list:
            for shm in shms:
                shm.unlink()
