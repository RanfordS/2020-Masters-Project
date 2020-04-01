# simple loop (i = 0, 1, ..., 4)
for i in range (3+2):
    print (i)

# function definition
# returns $(n!,\sum_{i=1}^{n} i!)$
def fact_and_sumf (n):
    # documentation string
    """Returns n! and sum(n!)"""
    fact = 1
    sumf = 0
    for i in range (1, n+1):
        fact *= i
        sumf += fact
    return fact, sumf

print (fact_and_sumf (4))
# (24, 33)

a = [2**i for i in [3,2,4]]
print (a)
# [8, 4, 16]

print (sum (a))
# 28
