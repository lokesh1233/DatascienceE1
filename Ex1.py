#White spaces and brackets

my_var = True
if my_var:
    print('my_var is True')

# No semicolons generally
# semicolons in python separate multiple statements on the same line
print('Hello'); print('World')

# String
my_str = 'This is string'

# Formatted strings
formatted_str = 'H{0}llo W{1}rld'.format('e', 'o')

# Data structures
my_list = [1, 2, 3]

for element in my_list:
    print(element)

# String in python work a bit like lists!
print(my_str[:3])

for token in my_str:
    print(token)

#List Comprehensions
divisible = [i for i in range(10) if i % 2 == 0 ]
print(divisible)

divisible = []
for i in range(10):
    if i % 2 == 0:
        divisible.append(i)

# Dictionaries 
# Key - value storage

d = {'bob' : '1234', 'jill' : '5678', 'julia' : '9999'} 
print(d['bob'])

for k, v in d.items():
    print(k, v)


# Dictionary Comprehension
new_numbers = {k:v for k, v in d.items() if k[0] == 'j'}
print(new_numbers)

# unordered collections with no duplicates
l = [1, 2, 2, 3, 4, 0]
print(set(l))

print(sorted(l))

# Comparing values
x = [1, 2, 3, 4]
y = [1, 2, 3, 4]
print(x == y)

# comparing objects
print(x is y)

print(x is None)
