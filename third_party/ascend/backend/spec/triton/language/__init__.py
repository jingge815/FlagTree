def language_extend_globals(globals_dict):
    from triton.tools.get_ascend_devices import is_compile_on_910_95
    globals_dict["is_compile_on_910_95"] = is_compile_on_910_95


def language_extend_globals_and_all(globals_dict, all_list):
    from .core import async_task
    globals_dict["async_task"] = async_task
    all_list.append("async_task")