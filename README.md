# 파이썬 공식사이트
## https://docs.python.org/ko/3/


# 자료구조

### Input
	# 1) input()을 호출해서 한 줄의 문자열을 입력받고, 그대로 변수 n에 저장
	n = input()  
	
	# 2) input()으로 문자열을 입력받은 뒤 int()로 정수로 변환하여 변수 n에 저장
	n = int(input())  
	
	# 3) input()으로 공백 구분된 문자열을 입력받고 split()으로 나눈 뒤,
	#    map(int, …)을 통해 각 토큰을 정수로 변환하는 map 객체를 생성하여 변수 n에 저장
	n = map(int, input().split())  
	
	# 4) input()으로 공백 구분된 문자열을 입력받고 split()으로 나눈 뒤,
	#    map(str, …)을 통해 각 토큰을 문자열로 변환하는 map 객체를 생성하여 변수 n에 저장
	n = map(str, input().split())  
	
	# 5) input()으로 공백 구분된 문자열을 입력받고 split()으로 나눈 뒤,
	#    map(int, …)으로 정수로 변환하고, list()로 감싸서 리스트로 만들어 변수 arr에 저장
	arr = list(map(int, input().split()))  
	
	# 6) for _ in range(N) 만큼 반복하며,
	#    각 줄마다 input()으로 공백 구분된 문자열을 입력받고 split() 후 map(int, …)으로 정수 리스트로 변환,
	#    그 리스트들을 모아 2차원 리스트 arr2를 생성
	arr2 = [list(map(int, input().split())) for _ in range(N)]

### 사칙연산
 	print(a / b) #나누기
  	print(a % b) #나머지
   	print(a // b) #몫
	print(a ** b) #거듭제곱

### 리스트
	arr=[]
	arr=[] * N
	arr.append(n)
	a.sort()
	a.sort(reverse=True)

### dict
	dict1={}

	# 밸류값 세기
 	from collections import Counter
	my_list = ['apple', 'banana', 'apple', 'cherry', 'banana', 'banana']
	count_dict = dict(Counter(my_list))
	print(count_dict)
	# 출력: {'apple': 2, 'banana': 3, 'cherry': 1}

	# 키밸류 변경
 	my_dict = {'apple': 1, 'banana': 2, 'cherry': 3}
	swapped_dict = {value: key for key, value in my_dict.items()}
	print(swapped_dict)
	# 출력: {1: 'apple', 2: 'banana', 3: 'cherry'}

### 큐
	from collections import deque
	d = deque()
	d.appendleft(0) #왼쪽에 데이터 삽입
	d.append(6) #오른쪽에 데이터삽입
	d.popleft() #왼쪽 데이터 지우기
	d.pop() #오른쪽 데이터 지우기

### 우선순위큐 (리스트에서 최소원소 추출하는 자료구조), 그리디 알고리즘에서 주로사용

 	import heapq
	heap = []
	heapq.heappush(heap, 50)
	heapq.heappush(heap, 10)
	heapq.heappush(heap, 20)
	print(heap)
 
# 알고리즘

### DFS
	def DFS(g,v,visited):
		visited[v] = True
		#print(v,end =' ')
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

### BFS
	from collections import deque
	
	def bfs(g,start,visited):
		queue = deque([start])
		visited[start] = True
		while queue:
			v = queue.popleft()
			#print(v,end=' ')
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

### 이진탐색
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

### 소수판별
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
	    

### 순열
	from itertools import permutations
	for i in permutations([1,2,3,4], 2):
    		print(i, end=" ")


### 조합 
	from itertools import combinations
	for i in combinations([1,2,3,4], 2):
    		print(i, end=" ")



