import inspect


def module_instance(cfg):
    module_name = cfg.get('name', None)
    if module_name is None:
        raise ValueError('Module name is required')
    args = cfg.get('args', {})
    namespace = cfg.get('namespace', [])
    if type(namespace) is str:
        namespace = [namespace]
    for ns in namespace:
        try:
            module = module_reflection(module_name, ns)
            if inspect.isclass(module):
                if type(args) is dict:
                    return module(**args)
                elif type(args) is list:
                    return module(*args)
                else:
                    return module(args)
            else:
                return module
        except ValueError:
            continue
    raise ValueError(f'Module {module_name} not found in {namespace}')


def module_reflection(module_name, namespaces):
    if type(namespaces) is str:
        namespaces = [namespaces]
    if module_name is None:
        raise ValueError('extra name is required')
    for ns in namespaces:
        try:
            module = getattr(__import__(ns, fromlist=['']), module_name)
            if callable(module):
                return module
        except AttributeError:
            continue
    raise ValueError(f'module {module_name} not found in {namespaces}')
