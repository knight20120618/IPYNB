# %% [markdown]
# 101

# %%
n1 = int(input())
n2 = int(input())
n3 = int(input())
n4 = int(input())
print('|{:>5d} {:>5d}|'.format(n1,n2))
print('|{:>5d} {:>5d}|'.format(n3,n4))
print('|{:<5d} {:<5d}|'.format(n1,n2))
print('|{:<5d} {:<5d}|'.format(n3,n4))

# %% [markdown]
# 102

# %%
n1 = float(input())
n2 = float(input())
n3 = float(input())
n4 = float(input())
print('|{:>7.2f} {:>7.2f}|'.format(n1,n2))
print('|{:>7.2f} {:>7.2f}|'.format(n3,n4))
print('|{:<7.2f} {:<7.2f}|'.format(n1,n2))
print('|{:<7.2f} {:<7.2f}|'.format(n3,n4))

# %% [markdown]
# 103

# %%
n1 = str(input())
n2 = str(input())
n3 = str(input())
n4 = str(input())
print('|{:>10s} {:>10s}|'.format(n1,n2))
print('|{:>10s} {:>10s}|'.format(n3,n4))
print('|{:<10s} {:<10s}|'.format(n1,n2))
print('|{:<10s} {:<10s}|'.format(n3,n4))

# %% [markdown]
# 104

# %%
import math
radius = eval(input())
print('Radius = {:.2f}'.format(radius))
perimeter = (radius*2)*math.pi
print('Perimeter = {:.2f}'.format(perimeter))
area = (radius**2)*math.pi
print('Area = {:.2f}'.format(area))

# %% [markdown]
# 105

# %%
height = eval(input())
width = eval(input())
perimeter = (height+width)*2
area = height*width
print('Height = {:.2f}'.format(height))
print('Width = {:.2f}'.format(width))
print('Perimeter = {:.2f}'.format(perimeter))
print('Area = {:.2f}'.format(area))

# %% [markdown]
# 106

# %%
x = eval(input())
y = eval(input())
z = eval(input())
speed = (z/1.6)/(x/60+y/3600)
print('Speed = {:.1f}'.format(speed))

# %% [markdown]
# 107

# %%
n1 = eval(input())
n2 = eval(input())
n3 = eval(input())
n4 = eval(input())
n5 = eval(input())
sums = n1+n2+n3+n4+n5
avg = sums/5
print(n1,n2,n3,n4,n5)
print('Sum = {:.1f}'.format(sums))
print('Average = {:.1f}'.format(avg))

# %% [markdown]
# 108

# %%
x1 = eval(input())
y1 = eval(input())
x2 = eval(input())
y2 = eval(input())
distance = (((x1-x2)**2)+((y1-y2)**2))**0.5
print('( {} , {} )'.format(x1,y1))
print('( {} , {} )'.format(x2,y2))
print('Distance = {:.4f}'.format(distance))

# %% [markdown]
# 109

# %%
import math
S = eval(input())
area = (5*(S**2))/(4*math.tan(math.pi/5))
print('Area = {:.4f}'.format(area))

# %% [markdown]
# 110

# %%
import math
N = eval(input())
S = eval(input())
area = (N*(S**2))/(4*math.tan(math.pi/N))
print('Area = {:.4f}'.format(area))

# %% [markdown]
# 201

# %%
even = int(input())
if even % 2 == 0:
  print('{} is an even number.'.format(even))
else:
  print('{} is not an even number.'.format(even))

# %% [markdown]
# 202

# %%
a = int(input())
if a % 15 == 0:
  print('{} is a multiple of 3 and 5.'.format(a))
elif a % 3 == 0:
  print('{} is a multiple of 3.'.format(a))
elif a % 5 == 0:
  print('{} is a multiple of 5.'.format(a))
else:
  print('{} is not a multiple of 3 or 5.'.format(a))

# %% [markdown]
# 203

# %%
leap = int(input())
if (leap % 4 == 0 and leap % 100 != 0) or (leap % 400 == 0):
  print('{} is a leap year.'.format(leap))
else:
  print('{} is not a leap year.'.format(leap))

# %% [markdown]
# 204

# %%
a = int(input())
b = int(input())
c = str(input())
if c == '+':
  d = a + b
  print(d)
elif c == '-':
  d = a - b
  print(d)
elif c == '*':
  d = a * b
  print(d)
elif c == '/':
  d = a / b
  print(d)
elif c == '//':
  d = a // b
  print(d)
elif c == '%':
  d = a % b
  print(d)

# %% [markdown]
# 205

# %%
x = str(input())
if x.isalpha():
  print('{} is an alphabet.'.format(x))
elif x.isdigit():
  print('{} is a number.'.format(x))
else:
  print('{} is a symbol.'.format(x))

# %% [markdown]
# 206

# %%
x = int(input())
if x >= 80 and x <= 100:
  print('A')
elif x < 80 and x >= 70:
  print('B')
elif x < 70 and x >= 60:
  print('B')
else:
  print('F')

# %% [markdown]
# 207

# %%
x = eval(input())
if x >= 38000:
  y = x * 0.7
  print(y)
elif x < 38000 and x >= 28000:
  y = x * 0.8
  print(y)
elif x < 28000 and x >= 18000:
  y = x * 0.9
  print(y)
elif x < 18000 and x >= 8000:
  y = x * 0.95
  print(y)

# %% [markdown]
# 208

# %%
num = str(input())
if num == '10':
  print('A')
elif num == '11':
  print('B')
elif num == '12':
  print('C')
elif num == '13':
  print('D')
elif num == '14':
  print('E')
elif num == '15':
  print('F')
else:
  print(num)

# %% [markdown]
# 209

# %%
x = float(input())
y = float(input())
z = (((x-5)**2)+((y-6)**2))**0.5
if z <= 15:
  print('Inside')
else:
  print('Outside')

# %% [markdown]
# 210

# %%
a = int(input())
b = int(input())
c = int(input())
if a+b > c and a+c > b and b+c > a:
  d = a+b+c
  print(d)
else:
  print('Invalid')

# %% [markdown]
# 301

# %%
a = int(input())
b = int(input())
total = 0
for i in range(a, b+1):
  total += i
print(total)

# %% [markdown]
# 302

# %%
a = int(input())
b = int(input())
total = 0
if a % 2 == 0:
  for i in range(a, b+1, 2):
    total += i
  print(total)
else:
  for i in range(a+1, b+1, 2):
    total += i
  print(total)

# %% [markdown]
# 303

# %%
x = int(input())
for i in range(1, x+1):
  for j in range(1, i+1):
    print('{:>4}'.format(i*j),end='')
  print('')

# %% [markdown]
# 304

# %%
a = int(input())
total = 0
for i in range(1, a+1):
  if i % 5 == 0:
    total += i
print(total)

# %% [markdown]
# 305

# %%
x = input()
print(x[::-1])

# %% [markdown]
# 306

# %%
n = int(input())
total = 1
for i in range(1, n+1):
  total *= i
print(total)

# %% [markdown]
# 307

# %%
n = int(input())
for i in range(1, n+1):
  for j in range(1, n+1):
    print('{:<2}* {:<2}= {:<4}'.format(j, i, i*j), end='')
  print()

# %% [markdown]
# 308

# %%
n = int(input())
for i in range(n):
  x = input()
  total = []
  for j in x:
    total.append(int(j))
  print('Sum of all digits of {} is {}'.format(x, sum(total)))

# %% [markdown]
# 309

# %%
money = int(input())
year = eval(input())
month = int(input())
print('{} \t  {}'.format('Month','Amount'))
total = money
for i in range(1, month+1):
  total = total + total*year/1200
  print('{:3d} \t {:.2f}'.format(i, total))

# %% [markdown]
# 310

# %%
n = int(input())
total = 0
for i in range(1, n):
  total += 1 / ((i**0.5) + ((i+1)**0.5))
print('{:.4f}'.format(total))

# %% [markdown]
# 401

# %%
num = []
for i in range(10):
  num.append(eval(input()))
print(min(num))

# %% [markdown]
# 402

# %%
num = []
while True:
  nums = eval(input())
  if nums == 9999:
    break
  num.append(nums)
print(min(num))

# %% [markdown]
# 403

# %%
a = int(input())
b = int(input())
count = 0
total = 0
for i in range(a, b+1):
  if i % 4 == 0 or i % 9 == 0:
    count += 1
    total += i
    print('{:<4}'.format(i), end='')
    if count % 10 == 0:
      print()
print('\n{}'.format(count))
print(total)

# %% [markdown]
# 404

# %%
num = input()
print(num[::-1])

# %% [markdown]
# 405

# %%
while True:
  num = int(input())
  if num == -9999:
    break
  elif num <= 100 and num >= 90:
    print('A')
  elif num <= 89 and num >= 80:
    print('B')
  elif num <= 79 and num >= 70:
    print('C')
  elif num <= 69 and num >= 60:
    print('D')
  else:
    print('E')

# %% [markdown]
# 406

# %%
while True:
  cm = eval(input())
  if cm == -9999:
    break
  else:
    kg = eval(input())
    bmi = kg / ((cm / 100)**2)
    print('BMI: {:.2f}'.format(bmi))
    if bmi < 18.5:
      print('State: {}'.format('under weight'))
    elif bmi < 25 and bmi >= 18.5:
      print('State: {}'.format('normal'))
    elif bmi < 30 and bmi >= 25:
      print('State: {}'.format('over weight'))
    elif bmi >= 30:
      print('State: {}'.format('fat'))

# %% [markdown]
# 407

# %%
while True:
  year = int(input())
  if year == -9999:
    break
  elif (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
    print('{} is a leap year.'.format(year))
  else:
    print('{} is not a leap year.'.format(year))

# %% [markdown]
# 408

# %%
even = 0
odd = 0
for i in range(10):
  num = int(input())
  if num % 2 == 0:
    even += 1
  else:
    odd += 1
print('Even numbers: {}'.format(even))
print('Odd numbers: {}'.format(odd))

# %% [markdown]
# 409

# %%
Nami = Chopper = X = 0
for i in range(5):
  n = int(input())
  if n == 1:
    Nami += 1
  elif n == 2:
    Chopper += 1
  else:
    X += 1
  print('Total votes of No.1: Nami =  {}'.format(Nami))
  print('Total votes of No.2: Chopper =  {}'.format(Chopper))
  print('Total null votes =  {}'.format(X))
if Nami > Chopper:
  print('=> No.1 Nami won the election.')
elif Nami < Chopper:
  print('=> No.2 Chopper won the election.')
else:
  print('=> No one won the election.')

# %% [markdown]
# 410

# %%
n = int(input())
for i in range(1,n+1):
  for j in range(n-i):
    print(' ', end='')
  for k in range(2*i-1):
    print('*', end='')
  print()

# %% [markdown]
# 501

# %%
department = str(input())
student = str(input())
name = str(input())
def compute(department, student, name):
  print('Department: {}'.format(department))
  print('Student ID: {}'.format(student))
  print('Name: {}'.format(name))
compute(department, student, name)

# %% [markdown]
# 502

# %%
x = int(input())
y = int(input())
def compute(x, y):
  z = x*y
  print(z)
compute(x, y)

# %% [markdown]
# 503

# %%
a = int(input())
b = int(input())
total = 0
def compute(a, b, total):
  for i in range(a, b+1):
    total += i
  print(total)
compute(a, b, total)

# %% [markdown]
# 504

# %%
a = int(input())
b = int(input())
def compute(a, b):
  c = a ** b
  print(c)
compute(a, b)

# %% [markdown]
# 505

# %%
a = str(input())
x = int(input())
y = int(input())
def compute(a, x, y):
  for i in range(y):
    for j in range(x):
      print('{} '.format(a), end='')
    print()
compute(a, x, y)

# %% [markdown]
# 506

# %%
a = int(input())
b = int(input())
c = int(input())
def compute(a, b ,c):
  d = b**2-4*a*c
  if d > 0:
    s1 = (-b+(b**2-4*a*c)**0.5)/(2*a)
    s2 = (-b-(b**2-4*a*c)**0.5)/(2*a)
    print('{}, {}'.format(s1, s2))
  elif d == 0:
    print(-b/(2*a))
  else:
    print('Your equation has no root.')
compute(a,b,c)

# %% [markdown]
# 507

# %%
def compute(x):
  n = True
  if x <= 1:
    n = False
  else:
    for i in range(2, x):
      if x % i == 0:
        n = False
        break
  return n
x = int(input())
if compute(x):
  print('Prime')
else:
  print('Not Prime')

# %% [markdown]
# 508

# %%
import math
(x,y) = eval(input())
def compute(x,y):
  g = math.gcd(x,y)
  print(g)
compute(x,y)

# %% [markdown]
# 509

# %%
import math
(x, y) = eval(input())
(m, n) = eval(input())
p = x*n+y*m
q = y*n
def compute(p, q):
  g = math.gcd(p,q)
  print('{}/{} + {}/{} = {}/{}'.format(x,y,m,n,int(p/g),int(q/g)))
compute(p, q)

# %% [markdown]
# 510

# %%
def compute(i):
  if i <= 1:
    return i
  else:
    return(compute(i-2)+compute(i-1))
n = int(input())
for i in range(n):
  print(compute(i),end=' ')

# %% [markdown]
# 601

# %%
num = []
for i in range(12):
  n = int(input())
  print('{:>3}'.format(n), end='')
  if i % 2 == 0:
    num.append(n)
  if i % 3 == 2:
    print('')
print(sum(num))

# %% [markdown]
# 602

# %%
total = 0
for i in range(5):
  n = input()
  if n == 'J':
    total += 11
  elif n == 'Q':
    total += 12
  elif n == 'K':
    total += 13
  elif n == 'A':
    total += 1
  else:
    total += int(n)
print(total)

# %% [markdown]
# 603

# %%
n = []
for i in range(10):
  n.append(int(input()))
num = sorted(n)
print(num[-1],num[-2],num[-3])

# %% [markdown]
# 604

# %%
n = []
total = 1
for i in range(10):
  n.append(int(input()))
  num = n.count(n[i])
  if num > total:
    total = num
    nums = n[i]
print(nums)
print(total)

# %% [markdown]
# 605

# %%
num = []
for i in range(10):
  num.append(int(input()))
nums = sorted(num)
total = sum(nums) - nums[0] - nums[9]
avg = total / 8
print(total)
print('{:.2f}'.format(avg))

# %% [markdown]
# 606

# %%
def compute(rows,cols):
  for i in range(rows):
    for j in range(cols):
      print('{:4}'.format(j-i), end='')
    print('')
rows = int(input())
cols = int(input())
compute(rows,cols)

# %% [markdown]
# 607

# %%
n = ['1st', '2nd', '3rd']
s = []
for i in range(len(n)):
  print('The {} student:'.format(n[i]))
  s.append([])
  for j in range(5):
    s[i].append(int(input()))
for k in range(3):
  print('Student {}'.format(k+1))
  print('#Sum {}'.format(sum(s[k])))
  print('#Average {:.2f}'.format(sum(s[k])/5))

# %% [markdown]
# 608

# %%
n = []
for i in range(9):
  n.append(int(input()))
max_ind = n.index(max(n))
min_ind = n.index(min(n))
max_x = max_ind //3
max_y = max_ind %3
min_x = min_ind //3
min_y = min_ind %3
print('Index of the largest number {} is: ({}, {})'.format(max(n),max_x,max_y))
print('Index of the smallest number {} is: ({}, {})'.format(min(n),min_x,min_y))

# %% [markdown]
# 609

# %%
print('Enter matrix 1:')
m1 = [[],[]]
for i in range(2):
  for j in range(2):
    print('[{}, {}]: '.format(i+1,j+1),end='')
    m1[i].append(eval(input()))
print('Enter matrix 2:')
m2 = [[],[]]
for i in range(2):
  for j in range(2):
    print('[{}, {}]: '.format(i+1,j+1),end='')
    m2[i].append(eval(input()))
print('Matrix 1:')
for i in range(2):
  for j in range(2):
    print('{} '.format(m1[i][j]),end='')
  print()
print('Matrix 2:')
for i in range(2):
  for j in range(2):
    print('{} '.format(m2[i][j]),end='')
  print()
print('Sum of 2 matrices:')
for i in range(2):
  for j in range(2):
    print('{} '.format(m1[i][j]+m2[i][j]),end='')
  print()

# %% [markdown]
# 610

# %%
n = []
for w in range(1, 5):
  print('Week {}:'.format(w))
  for D in range(1, 4):
    x = eval(input('Day {}:'.format(D)))
    n.append(x)
A = sum(n) / len(n)
print('Average: {:.2f}'.format(A))
print('Highest: {}'.format(max(n)))
print('Lowest: {}'.format(min(n)))

# %% [markdown]
# 701

# %%
num = []
while True:
  n = int(input())
  if n == -9999:
    break
  num.append(n)
nums = tuple(num)
print(nums)
print('Length: {}'.format(len(nums)))
print('Max: {}'.format(max(nums)))
print('Min: {}'.format(min(nums)))
print('Sum: {}'.format(sum(nums)))

# %% [markdown]
# 702

# %%
x = []
y = []
print('Create tuple1:')
while True:
  n = eval(input())
  if n == -9999:
    break
  x.append(n)
print('Create tuple2:')
while True:
  n = eval(input())
  if n == -9999:
    break
  y.append(n)
z = x + y
print('Combined tuple before sorting: {}'.format(tuple(z)))
print('Combined list after sorting: {}'.format(sorted(z)))

# %% [markdown]
# 703

# %%
nums = []
while True:
  n = str(input())
  if n == 'end':
    break
  nums.append(n)
nums = tuple(nums)
print(nums)
print(nums[0:3])
print(nums[-3:])

# %% [markdown]
# 704

# %%
num = []
while True:
  n = int(input())
  if n == -9999:
    break
  num.append(n)
nums = set(num)
print('Length: {}'.format(len(nums)))
print('Max: {}'.format(max(nums)))
print('Min: {}'.format(min(nums)))
print('Sum: {}'.format(sum(nums)))

# %% [markdown]
# 705

# %%
n1 = []
n2 = []
n3 = []
print('Input to set1:')
for i in range(5):
  n = int(input())
  n1.append(n)
N1 = set(n1)
print('Input to set2:')
for i in range(3):
  n = int(input())
  n2.append(n)
N2 = set(n2)
print('Input to set3:')
for i in range(9):
  n = int(input())
  n3.append(n)
N3 = set(n3)
print('set2 is subset of set1: {}'.format(N2.issubset(N1)))
print('set3 is superset of set1: {}'.format(N3.issuperset(N1)))

# %% [markdown]
# 706

# %%
n = int(input())
for i in range(n):
  word = input()
  SET = set(word.lower())
  if ' ' in SET:
    SET.remove(' ')
  if len(SET) >= 26:
    print('True')
  else:
    print('False')

# %% [markdown]
# 707

# %%
print("Enter group X's subjects:")
x = []
while True:
  n = str(input())
  if n == 'end':
    break
  x.append(n)
X = set(x)
print("Enter group Y's subjects:")
y = []
while True:
  n = str(input())
  if n == 'end':
    break
  y.append(n)
Y = set(y)
print(sorted(X.union(Y)))
print(sorted(Y.intersection(X)))
print(sorted(Y.difference(X)))
print(sorted(Y.symmetric_difference(X)))

# %% [markdown]
# 708

# %%
x = {}
y = {}
print('Create dict1:')
while True:
  key = input('Key: ')
  if key == 'end':
    break
  x[key] = input('Value: ')
print('Create dict2:')
while True:
  key = input('Key: ')
  if key == 'end':
    break
  y[key] = input('Value: ')
x.update(y)
for i in sorted(x.keys()):
  print('{}: {}'.format(i,x[i]))

# %% [markdown]
# 709

# %%
color_dict = {}
while True:
  key = input('Key: ')
  if key == 'end':
    break
  color_dict[key] = input('Value: ')
for i in sorted(color_dict.keys()):
  print('{}: {}'.format(i,color_dict[i]))

# %% [markdown]
# 710

# %%
x = {}
while True:
  key = input('Key: ')
  if key == 'end':
    break
  x[key] = input('Value: ')
y = input('Search key: ')
print(y in x.keys())

# %% [markdown]
# 801

# %%
word = str(input())
for i in range(len(word)):
  print("Index of '{}': {}".format(word[i], i), end='')
  print()

# %% [markdown]
# 802

# %%
word = str(input())
total = 0
for i in range(len(word)):
  total += ord(word[i])
  print("ASCII code for '{}' is {}".format(word[i], ord(word[i])))
print(total)

# %% [markdown]
# 803

# %%
word = input().split(' ')
print(word[-3], word[-2], word[-1])

# %% [markdown]
# 804

# %%
n = input()
print(n.upper())
print(n.title())

# %% [markdown]
# 805

# %%
n = str(input())
print('|{:<10}|'.format(n))
print('|{:^10}|'.format(n))
print('|{:>10}|'.format(n))

# %% [markdown]
# 806

# %%
word = input()
n = input()
def compute(word, n):
  total = 0
  for i in range(len(word)):
    if n == word[i]:
      total += 1
  print('{} occurs {} time(s)'.format(n, total))
compute(word, n)

# %% [markdown]
# 807

# %%
n = input().split(' ')
total = 0
for i in n:
  total += int(i)
avg = total / len(n)
print('Total = {}'.format(total))
print('Average = {:.1f}'.format(avg))

# %% [markdown]
# 808

# %%
word = input().split('-')
n = 0
for i in word:
  if i.isdigit():
    n += 1
if n == 3:
  print('Valid SSN')
else:
  print('Invalid SSN')

# %% [markdown]
# 809

# %%
word = input()
n = 0
for i in word:
  if i.isupper():
    n += 1
if len(word) < 8:
  print('Invalid password')
elif n < 1:
  print('Invalid password')
elif not word.isalnum():
  print('Invalid password')
else:
  print('Valid password')

# %% [markdown]
# 810

# %%
n = int(input())
for i in range(n):
  word = input().split(' ')
  word = [eval(j) for j in word]
  x = max(word) - min(word)
  print('{:.2f}'.format(x))

# %% [markdown]
# 901

# %%
files = open('write.txt', 'w', encoding='UTF-8')
for i in range(5):
  files.write(input() + '\n')
files.close()

# %% [markdown]
# 902

# %%
files = open('read.txt', 'r', encoding='UTF-8')
total = 0
num = files.read().split(' ')
for i in range(len(num)):
  total += int(num[i])
print(total)
files.close()

# %% [markdown]
# 903

# %%
files = open('data.txt', 'a+', encoding='UTF-8')
for i in range(5):
  files.write('\n' + input())
print('Append completed!')
print('Content of "data.txt":')
files.seek(0,0)
print(files.read())
files.close()

# %% [markdown]
# 904

# %%
files = open('read.txt', 'r', encoding='UTF-8')
name = []
high = []
weight = []
for i in files:
  print(i)
  num = i.split(' ')
  name.append(num[0])
  high.append(eval(num[1]))
  weight.append(eval(num[2]))
  h_max = high.index(max(high))
  w_max = weight.index(max(weight))
print('Average height: {:.2f}'.format(sum(high)/len(name)))
print('Average weight: {:.2f}'.format(sum(weight)/len(name)))
print('The tallest is {} with {:.2f}cm'.format(name[h_max],max(high)))
print('The heaviest is {} with {:.2f}kg'.format(name[w_max],max(weight)))
files.close()

# %% [markdown]
# 905

# %%
f = input()
word = input()
files = open(f, 'r', encoding='UTF-8')
text = files.read()
print('=== Before the deletion')
print(text)
text = text.replace(word,'')
print('=== After the deletion')
print(text)
files = open(f, 'w', encoding='UTF-8')
files.write(text)
files.close()

# %% [markdown]
# 906

# %%
f = input()
word0 = input()
word1 = input()
files = open(f, 'r', encoding='UTF-8')
text = files.read()
print('=== Before the replacement')
print(text)
text = text.replace(word0, word1)
print('=== After the replacement')
print(text)
files.close()

# %% [markdown]
# 907

# %%
f = input()
files = open(f, 'r', encoding='UTF-8')
line = word = char = 0
for i in files:
  line += 1
  word += len(i.split())
  char += sum([len(j) for j in i.split()])
print('{} line(s)'.format(line))
print('{} word(s)'.format(word))
print('{} character(s)'.format(char))
files.close()

# %% [markdown]
# 908

# %%
f = input()
n = int(input())
files = open(f, 'r', encoding='UTF-8')
data = sorted(files.read().split())
for i in sorted(set(data)):
  if data.count(i) == n:
    print(i)
files.close()

# %% [markdown]
# 909

# %%
files = open('data.dat', 'w', encoding='UTF-8')
for i in range(5):
  files.write(input() + '\n')
print('The content of "data.dat":')
files = open('data.dat', 'r', encoding='UTF-8')
for i in files:
  print(i)
files.close()

# %% [markdown]
# 910

# %%
f = m = 0
files = open('read.dat', 'r', encoding='UTF-8')
for i in files:
  print(i)
  num = i.split()
  if num[2] == '0':
    f += 1
  elif num[2] == '1':
    m += 1
print('Number of males: {}'.format(m))
print('Number of females: {}'.format(f))


