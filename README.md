# 파이썬 코딩테스트
### 1. 입력값 처리 (python 기준) 
#### (1). 구분자가 줄바꿈일 경우  
	n = int(input())
	data = []
	for i in range(n) :
	  data.append(int(input())) 
	 
	data.sort(reverse = True)   
	print(data)

#### (2). 구분자가 띄어쓰기일 경우
	m = int(input())
	
	data = list(map(int, input().split()))
	data.sort(reverse = True)
	print(data) 

#### (3). 공백을 기준으로 구분하여 적은 수의 데이터 입력 
	n, m, k = map(int, input().split())
	print(n,m,k)

#### (4). 이차원 배열 입력 초기화
	n = int(input())
	m = int(input()) 

	arr = []
	for i in range(n):
		arr.append(list(map(int, input().split())))
	
	print(arr)

### 2. 자료구조
#### (1). 수자료형
##### 나누기(실수형으로 처리)
	print(a / b)

##### 나머지
	print(a % b)

##### 몫
	print(a // b)

##### 거듭제곱
	print(a ** b)

##### 제곱근
	print(a ** 0.5)


#### (2). 리스트

##### 크기가 N이고 모든 값이 0인 1차원 리스트 초기화
	n = 10
	a = [0] * n
        [0,0,0,0,0,0,0,0,0,0]

##### 리스트 관련 메소드
	a = [1,4,3]
	print("기본 리스트 : ",a)

##### 리스트에 원소삽입
	a.append(2)
	print("삽입 : ",a)

##### 오름차운 정렬
	a.sort()
	print("오름차순 정렬 : ",a)

##### 내림차순 정렬
	a.sort(reverse = True)
	print("내림차순 정렬 : ",a)

##### 변수를 변경하지 않고 내장 함수를 사용해 정렬된 리스트 반환
	b = [2,3,6,1,4]
	result = sorted(b)

	print(b)
	print(result)

##### 리스트 원소 뒤집기
	a.reverse()
	print("원소 뒤집기 : ",a)

##### 특정한 인덱스에 데이터 추가
	a.insert(2,3)
	print("인덱스2에 3추가 : ",a)

##### 특정 값인 데이터 개수 세기
	print("값이 3인 데이터 개수 : ",a.count(3))

##### (맨 앞의)특정 값 데이터 삭제
	a.remove(3)
	print("값이 3인 데이터 삭제 : ",a)

##### 리스트 배열을 문자열로 바꾸기
	a = []
	string = ''.join(a)

##### 리스트 인덱스, 원소 같이 출력
	for i, letter in enumerate(['A', 'B', 'C'], start=1):
	    print(i, letter)
	'''1 A
	2 B
	3 C'''

##### 리스트 인덱스
	a = [1,2,3,4,5,6,7,8,9]
	
	a[-1] # 9
	a[-3] # 7
	a[1:4] # [2,3,4]
	
	# 0부터 19까지의 수 중에서 홀수만 포함하는 리스트
	array = [i for i in range(20) if i % 2 == 1]


#### (3). MAP

##### Key, Value 쌍 얻기(items)
	dic = dict()
	answer = [k for k, v in dic.items() if v == 1] 

##### Key 리스트 만들기(keys)
	a = {'name': 'pey', 'phone': '0119993323', 'birth': '1118'}
	a.keys()

##### Value 리스트 만들기(values)
	a.values()
	
	#value로 키 찾기
	aa = {'0': 'AA', 
	      '1': 'BB', 
	      '2': 'CC'}
	[k for k, v in aa.items() if v == 'CC']

##### 딕셔너리안에 키 값 있는지 찾기
	car = {"name" : "BMW", "price" : "7000"}
	
	if "name" in car:    
		print("Key exist! The value is " + car["name"])
	else:    
		print("Key not exist!")

##### 딕셔너리 컨프리헨션
	d = {e:[] for e in set(genres)}
	d[e[0]].append([e[1] , e[2]])

#### (4). 집합자료형
##### 중복없음, 순서없음
	s1 = set([1,2,3,4,5,6])
	s2 = set([4,5,6,7,8,9])
	s3 = s1 & s2
	print(s3)   #result : {4,5,6}
	s3 = s1 | s2
	print(s3)   #result : {1, 2, 3, 4, 5, 6, 7, 8, 9}
	s3 = s1 - s2
	print(s3)   #result : {1, 2, 3}
	s3 = s2 - s1
	print(s3)   #result : {8, 9, 7}


#### (5). 스택

##### (DFS -> 재귀함수)
	stack = []
	data = "tempData"
	stack.append(data)
	stack.pop()
	stack[-1] #top위치에 있는 데이터를 단순 확인


#### (6). 큐

##### (BFS) 주로 deque로 사용
	from collections import deque
	d = deque()
	d.appendleft(0) #왼쪽에 데이터 삽입
	d.append(6) #오른쪽에 데이터삽입
	d.popleft() #왼쪽 데이터 지우기
	d.pop() #오른쪽 데이터 지우기

#### (7). 우선순위큐

##### heap사용 (리스트에서 최소원소 추출하는 자료구조), 그리디 알고리즘에서 주로사용
	import heapq
	heap = []
	heapq.heappush(heap, 50)
	heapq.heappush(heap, 10)
	heapq.heappush(heap, 20)
	print(heap)

##### heappop 함수는 가장 작은 원소를 힙에서 제거함과 동시에 그를 결과값으로 리턴한다.
	result = heapq.heappop(heap)
	print(result)
	print(heap)


### 3. 알고리즘


#### (1). 정렬알고리즘
	a = []
	a.reverse() 

##### 기본값 오름차순 정렬
	a = [1, 10, 5, 7, 6]
	a.sort(reverse=True)

##### 새로운 그릇에 담고 정렬
	x = [1 ,11, 2, 3]
	y = sorted(x)
	print(y)


#### (2). DFS (재귀함수)
	# g = graph
	# v = visit
	# visited = visited
	def DFS(g,v,visited):
		visited[v] = True
		print(v,end =' ')
		for i in g[v]:
			if not visited[i]:
				dfs(g,i,visited)
	g =[
		[],
		[2,3,8],
		[1,7],
		[1,4,5],
		[3,5],
		[3,4],
		[7],
		[2,6,8],
		[1,7],
	]
	
	visited = [False] * 9
	
	dfs(g,1,visited)
	
	===>
	1 2 7 6 8 3 4 5

#### (2). BFS (큐함수)
	from collections import deque
	
	def bfs(g,start,visited):
		queue = deque([start])
		visited[start] = True
	
		while queue:
			v = queue.popleft()
			print(v,end=' ')
			for i in graph[v]:
				if not visited[i]:
					queue.append(i)
					visitied[i] = True
	
	g =[
		[],
		[2,3,8],
		[1,7],
		[1,4,5],
		[3,5],
		[3,4],
		[7],
		[2,6,8],
		[1,7],
	]
	visited = [False] * 9
	
	bfs(g,1,visited)
	
	==>
	1 2 3 8 7 4 5 6


#### (3). 이진탐색
	N = int(input())
	A = list(map(int, input().split()))
	A.sort()
	M = int(input())
	target_list = list(map(int, input().split()))
	for i in range(M):
	    find = False
	    target = target_list[i]
	    # 이진탐색 시작
	    start = 0
	    end = len(A) - 1
	    while start <= end:
	        midi = int((start + end) / 2)
	        midv = A[midi]
	        if midv > target:
	            end = midi - 1
	        elif midv < target:
	            start = midi + 1
	        else:
	            find = True
	            break
	    if find:
	        print(1)
	    else:
	        print(0)


#### (4). 파이썬이진탐색 함수 (정렬된 순서를 유지하면서 리스트 a에 데이터 x를 삽입할 가장 왼쪽/오른쪽 인덱스를 찾는 메서드)
	from bisect import bisect_left, bisect_right
	
	a = [1,2,4,4,6]
	x = 4
	
	print(bisect_left(a,x))
	print(bisect_right(a,x))


#### (5). 그리디 (그리디 최소값을 만드는 괄호 배치 찾기)
	answer = 0
	A = list(map(str, input().split("-")))
	
	def mySum(i):
	    sum = 0
	    temp = str(i).split("+")
	    for i in temp:
	        sum += int(i)
	    return sum
	
	for i in range(len(A)):
	    temp = mySum(A[i])
	    if i == 0:
	        answer += temp
	    else:
	        answer -= temp
	print(answer)
 

#### (6). 소수판별
	import math
	M, N = map(int, input().split())
	A = [0] * (N + 1)
	for i in range(2, N + 1):
	    A[i] = i
	for i in range(2, int(math.sqrt(N)) + 1):  # 제곱근까지만 수행
	    if A[i] == 0:
	        continue
	    for j in range(i + i, N + 1, i):  # 배수 지우기
	        A[j] = 0
	for i in range(M, N + 1):
	    if A[i] != 0:
	        print(A[i])
	    

#### (7). 순열
	from itertools import permutations
	for i in permutations([1,2,3,4], 2):
	    print(i, end=" ")


#### (8). 조합 
	from itertools import combinations
	for i in combinations([1,2,3,4], 2):
	    print(i, end=" ")


### 4. try except
try:
    st.pop()
except :
    return False
    

#### 2차원배열 정렬
	lst.sort(key=lambda x:x[0])

#### round() 함수의 예제
	round(3.14141414)
	3
