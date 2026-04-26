
def get_method_name(key: str) -> str:
    """Return formatted display name for a method key."""
    _method_names = {
    'blastp': 'BLASTp',
    'diamond': 'DIAMOND',
    'mmseq2': 'MMseqs2',
    'near': 'NEAR',
    'dctdomain': 'DCTdomain',
    'dhr': 'DHR',
    'plm': 'PLMSearch',
    'tmvec': 'TM-Vec',
    }
    return _method_names.get(key.lower(), key)

def get_color_set(set=0):
    if set == 0:
        return ['#e7cb94', '#e7ba52', '#e6ab02', '#86643f', '#c6dbef', '#9ecae1', '#6baed6', '#3182bd']
    elif set == 1:
        return ['#DD8452', '#4C72B0']
    elif set == 2:
        return ['#6baed6', '#e7ba52']
    else:
        raise ValueError("Invalid color set index. Valid values are 0, 1, and 2.")
