"""FFN model executor (PyTorch).

This module implements a client–server batch execution architecture for FFN inference.

Design rationale:
- FFN inference per subvolume is iterative (hundreds to thousands of forward passes)
- Multiple subvolumes can be segmented in parallel but share one GPU
- In client–server mode, multiple Canvas (client threads) submit requests to a queue;
  one server thread collects and runs batches
- This keeps the GPU busy even when batch_size=1 (clients take turns submitting)

Protocol:
- Input queue accepts:
  - Positive integer N: register client N
  - Negative integer (-N-1): unregister client N
  - "exit": request executor shutdown
  - Tuple (client_id, seed, image, fetches): inference request

- Each client has its own output queue for result dicts

Differences from original TF version:
- TF: session.run(feed_dict={...}) on static graph
- PyTorch: model(image, seed) directly, @torch.no_grad() disables gradients
- TF: I/O via feed_dict/fetch names
- PyTorch: I/O via tensors

Tensor layout conversion (in _schedule_batch):
- NumPy (BZYX1) → PyTorch (BCZYX): .transpose(0, 4, 1, 2, 3)
- PyTorch (BCZYX) → NumPy (BZYX1): .transpose(0, 2, 3, 4, 1)
"""

import _thread
import os
import queue
import threading
import time
from typing import Optional, Sequence

from absl import logging
import numpy as np
import torch

from ..training import model as ffn_model
from . import inference_utils
from .inference_utils import timer_counter


class TerminationException(Exception):
    pass


class ExecutorInterface:
    """Provides a client/server interface.

    Owns the communication channels and provides methods of communication
    with the server.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.outputs = {}
        self._input_queue = queue.Queue()
        self.exit_request = threading.Event()

    def queue_put(self, x):
        if self.exit_request.is_set():
            raise TerminationException()
        return self._input_queue.put(x)

    def queue_get(self, **kwargs):
        return self._input_queue.get(**kwargs)

    def get_output(self, client_id: int, timeout: int = 0):
        while True:
            try:
                return self.outputs[client_id].get(timeout=timeout)
            except queue.Empty:
                if self.exit_request.is_set():
                    raise TerminationException()


class ExecutorClient:
    """Client interface for the FFN executor."""

    def __init__(self, counters: inference_utils.Counters,
                 interface: ExecutorInterface):
        self._client_id = None
        self.counters = counters
        self._interface = interface

    def start(self) -> int:
        raise NotImplementedError()

    def finish(self):
        raise NotImplementedError()

    def predict(self, seed: np.ndarray, image: np.ndarray,
                fetches: Sequence[str]):
        raise NotImplementedError()


class ThreadingExecutorClient(ExecutorClient):
    """Client interface for a same-process executor."""

    def start(self) -> int:
        with self._interface.lock:
            if not self._interface.outputs:
                client_id = 0
            else:
                client_id = max(self._interface.outputs.keys()) + 1
            self._interface.outputs[client_id] = queue.Queue()

        self._interface.queue_put(client_id)
        self._client_id = client_id
        return client_id

    def finish(self):
        if self._client_id is None:
            return
        with self._interface.lock:
            del self._interface.outputs[self._client_id]
        self._interface.queue_put(-1 - self._client_id)

    def predict(self, seed: np.ndarray, image: np.ndarray,
                fetches: Sequence[str]):
        assert self._client_id is not None
        self._interface.queue_put((self._client_id, seed, image, fetches))
        with timer_counter(self.counters, 'client-wait'):
            return self._interface.get_output(self._client_id, timeout=1)


class BatchExecutor:
    """Base class for FFN executors."""

    def __init__(self, interface: ExecutorInterface,
                 model: ffn_model.FFNModel,
                 model_info: ffn_model.ModelInfo,
                 counters: inference_utils.Counters, batch_size: int):
        self._interface = interface
        self.model = model
        self.counters = counters
        self.batch_size = batch_size
        self.active_clients = 0
        self.registered_clients = set()

        # ModelInfo sizes are (x,y,z); convert to (z,y,x) for NumPy arrays
        self._input_seed_size = np.array(model_info.input_seed_size[::-1]).tolist()
        self._input_image_size = np.array(model_info.input_image_size[::-1]).tolist()
        self._pred_size = np.array(model_info.pred_mask_size[::-1]).tolist()

    def __del__(self):
        self.stop_server()

    def start_server(self):
        raise NotImplementedError()

    def stop_server(self):
        raise NotImplementedError()

    def get_client(self, subvol_counters):
        return ThreadingExecutorClient(subvol_counters, self._interface)

    def _run_executor(self):
        raise NotImplementedError()

    def _run_executor_log_exceptions(self):
        try:
            self._run_executor()
        except Exception as e:
            logging.exception(e)
            _thread.interrupt_main()
            time.sleep(10)
            os._exit(1)

    @property
    def num_devices(self):
        return 1


class ThreadingBatchExecutor(BatchExecutor):
    """Thread-based PyTorch batch executor.

    Server thread holds the model and GPU; client threads submit requests via a queue.

    Note: Number of clients can (and should) exceed batch_size. With batch_size=1,
    multiple clients keep the GPU busy by alternating submissions instead of
    leaving the GPU idle while one client prepares data.
    """

    def __init__(self,
                 interface: ExecutorInterface,
                 model: ffn_model.FFNModel,
                 model_info: ffn_model.ModelInfo,
                 counters: inference_utils.Counters,
                 batch_size: int,
                 expected_clients: int = 1):
        super().__init__(interface, model, model_info, counters, batch_size)

        self.total_clients = 0
        self.expected_clients = expected_clients

        self.input_seed = np.zeros(
            [batch_size] + self._input_seed_size + [1], dtype=np.float32)
        self.input_image = np.zeros(
            [batch_size] + self._input_image_size + [1], dtype=np.float32)
        self.th_executor = None

    def start_server(self):
        if self.th_executor is None:
            self.th_executor = threading.Thread(
                target=self._run_executor_log_exceptions)
            self._interface.exit_request.clear()
            self.th_executor.start()

    def stop_server(self):
        if self.th_executor is None:
            return
        logging.info('Requesting executor shutdown.')
        self._interface.queue_put('exit')
        self._interface.exit_request.set()
        self.th_executor.join()
        self.th_executor = None
        logging.info('Executor shutdown complete.')

    def _run_executor(self):
        """Main loop of the server thread which runs PyTorch inference."""
        logging.info('Executor starting, batch_size=%d.', self.batch_size)

        while self.active_clients or self.total_clients < self.expected_clients:
            self.counters.get(
                'executor-clients', cumulative=False).Set(self.active_clients)

            with timer_counter(self.counters, 'executor-input'):
                ready = []
                while (len(ready) < min(self.active_clients, self.batch_size) or
                       not self.active_clients):
                    try:
                        data = self._interface.queue_get(timeout=5)
                    except queue.Empty:
                        continue
                    if data == 'exit':
                        logging.info('Executor shut down requested.')
                        return
                    elif isinstance(data, int):
                        client_id = data
                        if client_id >= 0:
                            self.registered_clients.add(client_id)
                            self.total_clients += 1
                            self.active_clients += 1
                            logging.info('client %d starting', client_id)
                        else:
                            try:
                                self.registered_clients.remove(-client_id - 1)
                                logging.info('client %d terminating', -client_id - 1)
                                self.active_clients -= 1
                            except KeyError:
                                logging.warning(
                                    'client %d not known or already terminated',
                                    -client_id - 1)
                    else:
                        client_id, seed, image, fetches = data
                        l = len(ready)
                        self.input_seed[l, ..., 0] = seed
                        self.input_image[l, ..., 0] = image
                        ready.append(client_id)

            if ready:
                self._schedule_batch(ready, fetches)

        logging.info('Executor terminating.')

    @torch.no_grad()
    def _schedule_batch(self, client_ids: Sequence[int], fetches: Sequence[str]):
        """Run one batch of PyTorch inference.

        Format conversion:
        1. Take N requests from self.input_seed/image (BZYX1)
        2. Convert to PyTorch tensors (BCZYX)
        3. Run model inference
        4. Convert output back to NumPy (BZYX1)
        5. Dispatch to each client's output queue
        """
        with timer_counter(self.counters, 'executor-inference'):
            try:
                n = len(client_ids)
                # BZYX1 (numpy) -> BCZYX (pytorch)
                seed_t = torch.from_numpy(
                    self.input_seed[:n].transpose(0, 4, 1, 2, 3).copy()
                )
                image_t = torch.from_numpy(
                    self.input_image[:n].transpose(0, 4, 1, 2, 3).copy()
                )

                device = next(self.model.parameters()).device
                seed_t = seed_t.to(device)
                image_t = image_t.to(device)

                logits = self.model(image_t, seed_t)
                # BCZYX (pytorch) -> BZYX1 (numpy)
                logits_np = logits.cpu().numpy().transpose(0, 2, 3, 4, 1)

            except Exception as e:
                logging.exception(e)
                _thread.interrupt_main()
                raise e

        with timer_counter(self.counters, 'executor-output'):
            with self._interface.lock:
                for i, client_id in enumerate(client_ids):
                    try:
                        self._interface.outputs[client_id].put(
                            {'logits': logits_np[i, ...]})
                    except KeyError:
                        pass
