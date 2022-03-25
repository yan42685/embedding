list1 = [(1, 2), (2, 3), (3, 4)]
for (a, b) in list1:
    a = 2 + a - a
    b = 2 + b - b
    c = a + b
for (a, b) in list1:
    print(a, b)
