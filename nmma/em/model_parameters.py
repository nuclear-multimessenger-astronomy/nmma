import re


def Bu2022mv(data):

    # data_out = {}

    # Note, for this example, all phi's and theta's are the same
    # so we remove them from the list

    magkeys = data.keys()
    for jj, key in enumerate(magkeys):
        rr = re.findall(r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", key)
        print(rr)
