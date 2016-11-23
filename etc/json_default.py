def json_default(obj):
    if isinstance(obj, set):
        return list(obj)
    # convert numpy types such as np.ndarray, np.float64, np.int32, etc to equivalent regular types
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    if obj.__class__.__name__ == 'function':
        return 'function %s' % obj.__name__
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    return obj
