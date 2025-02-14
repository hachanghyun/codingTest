
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

 

# 자료구조
### 문제3 구간합구하기 - 11659 실버3
	import sys
	input = sys.stdin.readline
	suNo, quizNo = map(int, input().split())
	numbers = list(map(int, input().split()))
	prefix_sum = [0]
	temp = 0
	
	#합배열 만들기
	for i in numbers:
		temp += i
		prefix_sum.append(temp)
	
	#합배열에서 구간합 구하기
		s, e= map(int, input().split())
		print(prefix_sum[e] - prefix_sum[s-1])


### 문제8 좋은수구하기 1253 골드4

	import sys
	input = sys.stdin.readline
	N = int(input())
	Result = 0
	A = list(map(int, input().split()))
	A.sort()
	for k in range(N):
	    find = A[k]
	    i = int(0)
	    j = int(N - 1)
	    while i < j:  # 투 포인터 알고리즘
	        if A[i] + A[j] == find:  # find는 서로 다른 두 수의 합 이어야 함을 체크
	            if i != k and j != k:
	                Result += 1
	                break
	            elif i == k:
	                i += 1
	            elif j == k:
	                j -= 1
	        elif A[i] + A[j] < find:
	            i += 1
	        else:
	            j -= 1
	print(Result)

### 문제10 최솟값 찾기 11003 플레티넘 슬라이딩 윈도우

	from collections import deque
	N, L = map(int, input().split())
	mydeque = deque()
	now = list(map(int, input().split()))
	
	#새로운 값이 들어올때마다 정렬대신 현재수보다 큰 값을 덱에서 제거해 시간 복잡도를 줄임
	for i in range(N):
	    while mydeque and mydeque[-1][0] > now[i]:
	        mydeque.pop()
	    mydeque.append((now[i],i))
	    if mydeque[0][1] <= i - L: # 범위에서 벗어난 값은 덱에서 제거
	        mydeque.popleft()
	    print(mydeque[0][0], end=' ')

# 정렬
### 문제15 수 정렬하기 2750 브론즈1

	N = int(input())
	A = [0]*N
	
	for i in range(N):
	    A[i] = int(input())
	
	for i in range(N-1):
	    for j in range(N-1-i):
	        if A[j] > A[j+1]:
	            temp = A[j]
	            A[j] = A[j+1]
	            A[j+1] = temp
	
	for i in range(N):
	    print(A[i])


# 탐색
### 문제23 연결 요소의 개수 구하기 11724 실버4

	import sys
	sys.setrecursionlimit(10000)
	input = sys.stdin.readline
	n, m = map(int, input().split())
	A = [[] for _ in range(n+1)]
	visited = [False] * (n+1)
	
	def DFS(v):
	    visited[v] = True
	    for i in A[v]:
	        if not visited[i]:
	            DFS(i)
	
	for _ in range(m):
	    s, e = map(int, input().split())
	    A[s].append(e)  # 양방향 에지이므로 양쪽에 에지를 더하기
	    A[e].append(s)
	count = 0
	for i in range(1, n+1):
	    if not visited[i]:  # 연결 노드 중 방문하지 않았던 노드만 탐색
	        count += 1
	        DFS(i)
	print(count)

### 문제26 DFS와 BFS 프로그램 1260 실버2

	from collections import deque
	
	N, M, Start = map(int, input().split())
	A = [[] for _ in range(N + 1)]
	for _ in range(M):
	    s, e = map(int, input().split())
	    A[s].append(e)  # 양방향 에지이므로 양쪽에 에지를 더하기
	    A[e].append(s)
	for i in range(N + 1):
	    A[i].sort()  # 번호가 작은 노드 부터 방문하기 위해 정렬하기
	visited = [False] * (N + 1)
	
	
	def DFS(v):
	    print(v, end=' ')
	    visited[v] = True
	    for i in A[v]:
	        if not visited[i]:
	            DFS(i)
	
	
	DFS(Start)
	
	visited = [False] * (N + 1)  # 리스트 초기화
	
	
	def BFS(v):
	    queue = deque()
	    queue.append(v)
	    visited[v] = True
	    while queue:
	        now_Node = queue.popleft()
	        print(now_Node, end=' ')
	        for i in A[now_Node]:
	            if not visited[i]:
	                visited[i] = True
	                queue.append(i)
	
	
	print()
	BFS(Start)

### 문제29 원하는 정수 찾기 1920 실버4

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

# 그리디
### 문제36 최솟값을 만드는 괄홓 배치 찾기 1541 실버2 

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

# 정수론
### 문제37 소수구하기 1929 실버3

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

# 그래프
### 문제50 집합 표현하기 1717 골드4
	import sys
	input = sys.stdin.readline
	sys.setrecursionlimit(100000)
	N, M = map(int, input().split())
	parent = [0] * (N + 1)
	
	
	def find(a):
	    if a == parent[a]:
	        return a
	    else:
	        parent[a] = find(parent[a])  # 재귀 형태로 구현 -> 경로 압축 부분
	        return parent[a]
	
	
	def union(a, b):
	    a = find(a)
	    b = find(b)
	    if a != b:
	        parent[b] = a
	
	
	def checkSame(a, b):
	    a = find(a)
	    b = find(b)
	    if a == b:
	        return True
	    return False
	
	
	for i in range(0, N + 1):
	    parent[i] = i
	
	for i in range(M):
	    question, a, b = map(int, input().split())
	    if question == 0:
	        union(a, b)
	    else:
	        if checkSame(a, b):
	            print("YES")
	        else:
	            print("NO")

### 문제54 게임 개발하기 1516 골드3

	from collections import deque
	
	N = int(input())
	A = [[] for _ in range(N + 1)]
	indegree = [0] * (N + 1)  # 진입차수 리스트
	selfBuild = [0] * (N + 1)  # 자기자신을 짓는데 걸리는 시간
	
	for i in range(1, N + 1):
	    inputList = list(map(int, input().split()))
	    selfBuild[i] = (inputList[0])  # 건물을 짓는데 걸리는 시간
	    index = 1
	    while True:  # 인접리스트 만들기
	        preTemp = inputList[index]
	        index += 1
	        if preTemp == -1:
	            break
	        A[preTemp].append(i)
	        indegree[i] += 1  # 진입차수 데이터 저장
	
	queue = deque()
	for i in range(1, N + 1):
	    if indegree[i] == 0:
	        queue.append(i)
	
	result = [0] * (N + 1)
	while queue:  # 위상정렬 수행
	    now = queue.popleft()
	    for next in A[now]:
	        indegree[next] -= 1
	        result[next] = max(result[next], result[now] + selfBuild[now])
	        if indegree[next] == 0:
	            queue.append(next)
	
	for i in range(1, N + 1):
	    print(result[i] + selfBuild[i])

### 문제56 최단 경로 구하기 1753 골드4

	import sys
	input = sys.stdin.readline
	from queue import PriorityQueue
	
	V, E = map(int, input().split())
	K = int(input())
	distance = [sys.maxsize] * (V + 1)
	visited = [False] * (V + 1)
	myList = [[] for _ in range(V + 1)]
	q = PriorityQueue()
	
	for _ in range(E):
	    u, v, w = map(int, input().split())  # 가중치가 있는 인접 리스트 저장
	    myList[u].append((v, w))
	
	q.put((0, K))  # K를 시작점으로 설정
	distance[K] = 0
	while q.qsize() > 0:
	    current = q.get()
	    c_v = current[1]
	    if visited[c_v]:
	        continue
	    visited[c_v] = True
	    for tmp in myList[c_v]:
	        next = tmp[0]
	        value = tmp[1]
	        if distance[next] > distance[c_v] + value:  # 최소 거리로 업데이트
	            distance[next] = distance[c_v] + value
	            q.put((distance[next], next))
	for i in range(1, V + 1):
	    if visited[i]:
	        print(distance[i])
	    else:
	        print("INF")

### 문제58 K번째 최단 경로 찾기 1854 플래티넘
	import sys
	import heapq
	input = sys.stdin.readline
	N, M, K = map(int, input().split())
	W = [[] for _ in range(N+1)]
	distance = [[sys.maxsize] * K for _ in range(N+1)]
	
	for _ in range(M):
	    a, b, c = map(int, input().split())
	    W[a].append((b,c))
	
	pq = [(0,1)]
	distance[1][0] = 0
	while pq:
	    cost,node = heapq.heappop(pq)
	    for nNode, nCost in W[node]:
	        sCost = cost + nCost
	        if distance[nNode][K-1] > sCost:
	            distance[nNode][K-1] = sCost
	            distance[nNode].sort()
	            heapq.heappush(pq, [sCost, nNode])
	
	for i in range(1,N+1):
	    if distance[i][K-1] == sys.maxsize:
	        print(-1)
	    else:
	        print(distance[i][K-1])

### 문제59 타임머신으로 빨리가기 11657 골드4

	import sys
	input = sys.stdin.readline
	N, M = map(int, input().split())
	edges = []
	distance = [sys.maxsize]*(N+1)
	for i in range(M):  # 에지 데이터 저장
	    start, end, time = map(int, input().split())
	    edges.append((start, end, time))
	
	# 벨만포드 수행
	distance[1] = 0
	for _ in range(N-1):
	    for start, end, time in edges:
	        if distance[start] != sys.maxsize and distance[end] > distance[start] + time:
	            distance[end] = distance[start] + time
	
	# 음수 사이클 확인
	mCycle = False
	for start, end, time in edges:
	    if distance[start] != sys.maxsize and distance[end] > distance[start] + time:
	        mCycle = True
	
	if not mCycle:
	    for i in range(2, N+1):
	        if distance[i] != sys.maxsize:
	            print(distance[i])
	        else:
	            print(-1)
	else:
	    print(-1)


### 문제61 가장 빠른 버스 노선 구하기 11404 골드4

	import sys
	input = sys.stdin.readline
	N = int(input())
	M = int(input())
	distance = [[sys.maxsize for j in range(N+1)] for i in range(N+1)]
	for i in range(1, N+1): # 인접 행렬 초기화
	    distance[i][i] = 0
	
	for i in range(M):
	    s, e, v = map(int, input().split())
	    if distance[s][e] > v:
	        distance[s][e] = v
	
	# 플로이드 워셜 수행
	for k in range(1, N+1):
	    for i in range(1, N+1):
	        for j in range(1, N+1):
	            if distance[i][j] > distance[i][k] + distance[k][j]:
	                distance[i][j] = distance[i][k] + distance[k][j]
	
	for i in range(1, N+1):
	    for j in range(1, N+1):
	        if distance[i][j] == sys.maxsize:
	            print(0, end=' ')
	        else:
	            print(distance[i][j], end=' ')
	    print()

### 문제64 최소 신장 트리 구하기 1197 골드4 


	import sys
	from queue import PriorityQueue
	
	input = sys.stdin.readline
	N, M = map(int, input().split())
	pq = PriorityQueue()
	parent = [0] * (N + 1)
	for i in range(N + 1):
	    parent[i] = i
	
	for i in range(M):
	    s, e, v = map(int, input().split())
	    pq.put((v, s, e))  # 제일 앞 순서로 정렬되므로 가중치를 제일 앞 순서로 함
	
	def find(a):
	    if a == parent[a]:
	        return a
	    else:
	        parent[a] = find(parent[a])
	        return parent[a]
	
	def union(a, b):
	    a = find(a)
	    b = find(b)
	    if a != b:
	        parent[b] = a
	
	useEdge = 0
	result = 0
	while useEdge < N - 1:  # MST는 한상 N-1의 에지를 사용함
	    v, s, e = pq.get()
	    if find(s) != find(e):  # 같은 부모가 아닌 경우만 연결
	        union(s, e)
	        result += v
	        useEdge += 1
	
	print(result)


# 트리
### 문제71 구간합 구하기3 2042 골드1

	import sys
	input = sys.stdin.readline
	N, M, K = map(int, input().split())  # 수의 개수, 변경이 일어나는 횟수, 구간 합을 구하는 횟수
	treeHeight = 0
	lenght = N
	
	while lenght != 0:
	    lenght //= 2
	    treeHeight += 1
	
	treeSize = pow(2, treeHeight + 1)
	leftNodeStartIndex = treeSize // 2 - 1
	tree = [0] * (treeSize + 1)
	
	# 데이터를 리프노드에 저장
	for i in range(leftNodeStartIndex + 1, leftNodeStartIndex + N + 1):
	    tree[i] = int(input())
	
	# 인덱스 트리 생성 함수
	def setTree(i):
	    while i != 1:
	        tree[i // 2] += tree[i]
	        i -= 1
	
	setTree(treeSize - 1)
	
	
	# 값 변경 함수
	def changeVal(index, value):
	    diff = value - tree[index]
	    while index > 0:
	        tree[index] = tree[index] + diff
	        index = index // 2
	
	# 구간 합 계산 함수
	def getSum(s, e):
	    partSum = 0
	    while s <= e:
	        if s % 2 == 1:
	            partSum += tree[s]
	            s += 1
	        if e % 2 == 0:
	            partSum += tree[e]
	            e -= 1
	        s = s // 2
	        e = e // 2
	    return partSum
	
	for _ in range(M + K):
	    question, s, e = map(int, input().split())
	    if question == 1:
	        changeVal(leftNodeStartIndex + s, e)
	    elif question == 2:
	        s = s + leftNodeStartIndex
	        e = e + leftNodeStartIndex
	        print(getSum(s, e))

### 문제75 최소 공통 조상 구하기 2 11438 플래티넘


	import sys
	from collections import deque
	input = sys.stdin.readline
	print = sys.stdout.write
	N = int(input())
	tree = [[0] for _ in range(N + 1)]
	
	for _ in range(0, N - 1):  # 인접 리스트에 트리 데이터 저장
	    s, e = map(int, input().split())
	    tree[s].append(e)
	    tree[e].append(s)
	
	depth = [0] * (N + 1)
	visited = [False] * (N + 1)
	temp = 1
	kmax = 0
	while temp <= N:  # 최대 가능 depth 구하기
	    temp <<= 1
	    kmax += 1
	
	parent = [[0 for j in range(N + 1)] for i in range(kmax + 1)]
	
	def BFS(node):
	    queue = deque()
	    queue.append(node)
	    visited[node] = True
	    level = 1
	    now_size = 1
	    count = 0
	    while queue:
	        now_node = queue.popleft()
	        for next in tree[now_node]:
	            if not visited[next]:
	                visited[next] = True
	                queue.append(next)
	                parent[0][next] = now_node  # 부모 노드 저장
	                depth[next] = level  # 노드 depth 저장
	        count += 1
	        if count == now_size:
	            count = 0
	            now_size = len(queue)
	            level += 1
	
	
	BFS(1)
	
	for k in range(1, kmax + 1):
	    for n in range(1, N + 1):
	        parent[k][n] = parent[k - 1][parent[k - 1][n]]
	
	def excuteLCA(a, b):
	    if depth[a] > depth[b]:  # 더 깊은 depth가 b가 되도록
	        temp = a
	        a = b
	        b = temp
	
	    for k in range(kmax, -1, -1):  # depth 빠르게 맞추기
	        if pow(2, k) <= depth[b] - depth[a]:
	            if depth[a] <= depth[parent[k][b]]:
	                b = parent[k][b]
	
	    for k in range(kmax, -1, -1):  # 조상 빠르게 찾기
	        if a == b: break
	        if parent[k][a] != parent[k][b]:
	            a = parent[k][a]
	            b = parent[k][b]
	
	    LCA = a
	    if a != b:
	        LCA = parent[0][LCA]
	    return LCA
	
	M = int(input())
	for _ in range(M):
	    a, b = map(int, input().split())
	    print(str(excuteLCA(a, b)))
	    print("\n")

# 조합
### 문제81 순열의 순서 구하기 1722 골드4

	import sys
	input = sys.stdin.readline
	
	F = [0]*21
	S = [0]*21
	visited = [False]*21
	N = int(input())
	
	F[0] = 1
	for i in range(1, N+1): # 팩토리얼 초기화 → 각 자릿수에서 만들 수 있는 경우의 수
	    F[i] = F[i-1] * i
	
	inputList = list(map(int, input().split()))
	
	if inputList[0] == 1:
	    K = inputList[1]
	    for i in range(1,N+1):
	        cnt = 1
	        for j in range(1, N+1):
	            if visited[j]:  # 이미 사용한 숫자는 사용할 수 없음
	                continue
	            if K <= cnt*F[N-i]: # 주어진 K에 따라 각 자리에 들어갈 수 있는 수 찾기
	                K -= (cnt-1) * F[N-i]
	                S[i] = j
	                visited[j] = True
	                break
	            cnt += 1
	    for i in range(1, N+1):
	        print(S[i], end=' ')
	
	else:
	    K = 1
	    for i in range(1, N+1):
	        cnt = 0
	        for j in range(1, inputList[i]):
	            if not visited[j]:
	                cnt += 1    # 미사용 숫자 갯수 만큼 카운트
	        K += cnt * F[N-i]   # 자릿수에 따라 순서 더하기
	        visited[inputList[i]] = True
	    print(K)

### 문제82 사전 찾기 1256 골드3

	import sys
	input = sys.stdin.readline
	N, M, K = map(int, input().split())
	D = [[0 for j in range(202)] for i in range(202)]
	
	for i in range(0, 201):
	    for j in range(0, i + 1):
	        if j == 0 or j == i:
	            D[i][j] = 1
	        else:
	            D[i][j] = D[i - 1][j - 1] + D[i - 1][j]
	            if D[i][j] > 1000000000:
	                D[i][j] = 1000000001
	
	if D[N + M][M] < K:
	    print(-1)
	else:
	    while not (N == 0 and M == 0):
	        if D[N - 1 + M][M] >= K:
	            print("a", end='')
	            N -= 1
	        else:
	            print("z", end='')
	            K -= D[N - 1 + M][M]
	            M -= 1

# 동적 계획법
### 문제86 이천수 구하기 2193 실버3

	import sys
	input = sys.stdin.readline
	N = int(input())
	D = [[0 for j in range(2)] for i in range(N+1)]
	D[1][1] = 1 # 1자리 이친수는 1 한 가지만 있음
	D[1][0] = 0
	
	for i in range(2, N+1):
	    D[i][0] = D[i-1][1] + D[i-1][0]
	    D[i][1] = D[i-1][0]
	
	print(D[N][0] + D[N][1])


### 문제90 최장 공통 부분 수열 찾기 9252 골드4

	import sys
	sys.setrecursionlimit(10000)
	input = sys.stdin.readline
	A = list(input())
	A.pop()  # \n 문자열 제거
	B = list(input())
	B.pop()  # \n 문자열 제거
	DP = [[0 for j in range(len(B) + 1)] for i in range(len(A) + 1)]
	Path = []
	for i in range(1, len(A) + 1):
	    for j in range(1, len(B) + 1):
	        if A[i - 1] == B[j - 1]:
	            DP[i][j] = DP[i - 1][j - 1] + 1  # 같은 문자열일 때 왼쪽 대각선 값 + 1
	        else:
	            DP[i][j] = max(DP[i - 1][j], DP[i][j - 1])  # 다르면 왼쪽과 위의 값 중 큰 수
	
	print(DP[len(A)][len(B)])
	
	
	# LCS 구현 함수
	def getText(r, c):
	    if r == 0 or c == 0:
	        return
	    if A[r - 1] == B[c - 1]:  # 같으면 LCS에 기록하고 왼쪽위로 이동
	        Path.append(A[r - 1])
	        getText(r - 1, c - 1)
	    else:  # 다르면 왼쪽과 중 큰 수로 이동
	        if DP[r - 1][c] > DP[r][c - 1]:
	            getText(r - 1, c)
	        else:
	            getText(r, c - 1)
	
	
	getText(len(A), len(B))
	
	for i in range(len(Path) - 1, -1, -1):
	    print(Path.pop(i), end='')
	print()


### 문제94 행렬 곱 연산 횟숫의 최솟값 구하기 11049 골드3

	import sys
	
	input = sys.stdin.readline
	N = int(input())
	M = []
	D = [[-1 for j in range(N + 1)] for i in range(N + 1)]
	
	M.append((0, 0))
	for i in range(N):
	    x, y = map(int, input().split())
	    M.append((x, y))
	
	def excute(s, e):
	    result = sys.maxsize
	    if D[s][e] != -1:
	        return D[s][e]
	    if s == e:
	        return 0
	    if s + 1 == e:
	        return M[s][0] * M[s][1] * M[e][1]
	    for i in range(s, e):
	        result = min(result, M[s][0] * M[i][1] * M[e][1] + excute(s, i) + excute(i + 1, e))
	    D[s][e] = result
	    return D[s][e]
	
	print(excute(1, N))

### 문제95 외판원의 순회 경로 짜기 2098 골드1

	import sys
	input = sys.stdin.readline
	N = int(input())
	W = []
	for i in range(N):
	 W.append([])
	 W[i] = list(map(int, input().split()))
	D = [[0 for j in range(1 << 16)] for i in range(16)]
	
	def tsp(c,v):
	    if v == (1<<N)-1:
	        if W[c][0] == 0:
	            return float('inf')
	        else:
	            return W[c][0]
	    if D[c][v] != 0:
	        return D[c][v]
	    min_val = float('inf')
	    for i in range(0, N):
	        if (v & (1<<i)) == 0 and W[c][i] != 0:
	            min_val = min(min_val, tsp(i, (v | (1 << i))) + W[c][i])
	    D[c][v] = min_val
	    return D[c][v]
	
	print(tsp(0,1))


### 문제96 가장 길게 증가하는 부분 수열 찾기 14003 플래티넘

	import sys
	input = sys.stdin.readline
	N = int(input())
	A = list(map(int, input().split()))
	A.insert(0, 0)
	
	index = 0
	maxLength = 1
	B = [0] * 1000001
	D = [0] * 1000001
	ans = [0] * 1000001
	B[maxLength] = A[1]
	D[1] = 1
	
	# 바이너리 서치 구현
	def binarysearch(l, r, now):
	    while l < r:
	        mid = (l + r) // 2
	        if B[mid] < now:
	            l = mid + 1
	        else:
	            r = mid
	    return l
	
	for i in range(2, N + 1):
	    if B[maxLength] < A[i]:  # 가장 마지막 수열보다 현재 수열이 큰 경우
	        maxLength += 1
	        B[maxLength] = A[i]
	        D[i] = maxLength
	    else:  # 바이너리 서치를 이용해 현재 수열이 들어갈 index 찾기
	        index = binarysearch(1, maxLength, A[i])
	        B[index] = A[i]
	        D[i] = index
	
	print(maxLength)
	index = maxLength
	x = B[maxLength] + 1
	for i in range(N, 0, -1):
	    if D[i] == index and A[i] < x:
	        ans[index] = A[i]
	        x = A[i]
	        index -= 1
	
	for i in range(1, maxLength + 1):
	    print(ans[i], end=' ')

