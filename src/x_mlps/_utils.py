from functools import partial

# Taken from https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py
# Lightly modified - all credit goes to lucidrains


def _string_begins_with(prefix, str):
    return str.startswith(prefix)


def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key, None), keys))
    return values


def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)


def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(_string_begins_with, prefix), d)


def group_by_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(_string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix) :], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs


__all__ = ["group_by_key_prefix", "group_by_prefix_and_trim", "group_dict_by_key", "pick_and_pop"]
