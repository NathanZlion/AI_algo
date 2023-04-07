


import random
#Generate 5 random numbers between 10 and 30

nonrepeating = True
for _ in range(100):
    randomlist = random.sample(range(10, 30), 5)
    if len(set(randomlist)) != len(randomlist):
        nonrepeating = False

print(nonrepeating)
