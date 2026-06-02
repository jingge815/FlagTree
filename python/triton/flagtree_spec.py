import os

# flagtree backend path specialization
def spec_path(path_list: list):
    from ._flagtree_backend import FLAGTREE_BACKEND
    if not path_list or not FLAGTREE_BACKEND:
        return
    current_path = path_list[0].replace(os.sep, "/")
    marker = "/triton/"
    idx = current_path.find(marker)
    if idx == -1:
        return
    triton_root = current_path[:idx + len("/triton")]
    rel_path = current_path[idx + len(marker):]
    backend_path = os.path.join(triton_root, "backends", FLAGTREE_BACKEND, "spec", "triton", rel_path)
    if os.path.isdir(backend_path) and backend_path not in path_list:
        path_list.insert(0, backend_path)


# flagtree backend specialization
def spec(function_name: str, *args, **kwargs):
    from .runtime.driver import driver
    if hasattr(driver.active, "spec"):
        spec = driver.active.spec
        if hasattr(spec, function_name):
            func = getattr(spec, function_name)
            return func(*args, **kwargs)
    return None


# flagtree backend func specialization
def spec_func(function_name: str):
    from .runtime.driver import driver
    if hasattr(driver.active, "spec"):
        spec = driver.active.spec
        if hasattr(spec, function_name):
            func = getattr(spec, function_name)
            return func
    return None