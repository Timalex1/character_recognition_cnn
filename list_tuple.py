
#!/usr/bin/python3
def complex_delete(a_dictionary, value):
    if a_dictionary is None:
        return

    id = []

    for i in a_dictionary:
        if a_dictionary[i] == value:
            id.append(i)

    if id != None:
        for i in id:
            del a_dictionary[i]

    return a_dictionary

