import dill

def run_dill_encoded(payload):
    fun, args = dill.loads(payload)
    return fun(*args)

def apply_async_dill(pool, fun, args, callback, error_callback=print):
    payload = dill.dumps((fun, args))
    return pool.apply_async(run_dill_encoded, (payload,), callback=callback, error_callback=error_callback)


