

def suffix_replace(original, old, new):
    """
    Replaces the old suffix of the original string by a new suffix
    """
    return original[: -len(old)] + new


def prefix_replace(original, old, new):
    """
     Replaces the old prefix of the original string by a new suffix
    :param original: string
    :param old: string
    :param new: string
    :return: string
    """
    return new + original[len(old) :]
