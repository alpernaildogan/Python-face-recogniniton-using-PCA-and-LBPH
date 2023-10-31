
lst = [1,1,1,2]

def most_common(lst):
    return max(set(lst), key=lst.count)

c = most_common(lst)
print (c)
