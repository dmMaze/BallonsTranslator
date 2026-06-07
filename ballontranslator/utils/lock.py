RUNTIME_LOCKS = {}


def aquire_model_loading_lock():
    lock = RUNTIME_LOCKS.get('model_loading', None)
    if lock is not None:
        lock.lock()


def release_model_loading_lock():
    lock = RUNTIME_LOCKS.get('model_loading', None)
    if lock is not None:
        lock.unlock()