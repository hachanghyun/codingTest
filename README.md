# 파이썬 공식사이트
## https://docs.python.org/ko/3/

## Library reference
### dict
### collections
	deque
 	Counter
  	defaultdict
### heapq
### bisect
### itertools

# 1.자료구조

## 1) Input
	# 한 줄 문자열 입력
	n = input()
	
	# 정수 입력
	n = int(input())
	
	# 공백 구분 정수를 map 객체로 입력
	n = map(int, input().split())
	
	# 공백 구분 문자열을 map 객체로 입력
	n = map(str, input().split())
	
	# 공백 구분 정수를 리스트로 입력
	arr = list(map(int, input().split()))
	
	# N줄 반복해 2차원 정수 리스트 생성
	arr2 = [list(map(int, input().split())) for _ in range(N)]

 	# 띄어쓰기 Input값 변수에 담기
  	time, numer, io = re.split()
   

## 2) 사칙연산
	# 나누기 /
 	print(a / b) 

  	# 나머지 %
  	print(a % b) 

   	# 몫 //
   	print(a // b)

	# 거듭제곱 **
	print(a ** b) 

## 3) 리스트, 배열
	# 리스트 초기화
	arr=[]
 	arr=[] * N

	# 2차원 리스트 초기화 (bfs, dfs)
 	visited = [[False for _ in range(m)] for _ in range(n)] n: 행개수, m: 열개수

  	# 리스트 원소 추가
	arr.append(n)

 	# 리스트 정렬
	a.sort()
	a.sort(reverse=True)

## 4) dict (key-value)
	# dict 초기화
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

 	# dict value 찾기
  	for k, v in dict1.items():
   		if v == '장미':
     			print(v)

 	# dict key정렬
  	sorted(dict1.items(), key=lambda x: x[0])
   	
 	# dict value정렬
  	sorted(dict1.items(), key=lambda x: x[1])

## 5) queue (선입선출)
	# queue 초기화
	from collections import deque
	d = deque()

 	# 왼쪽에 데이터 삽입
	d.appendleft(0)

 	# 오른쪽에 데이터삽입
	d.append(6)

  	# 왼쪽 데이터 지우기
	d.popleft()

 	# 오른쪽 데이터 지우기
	d.pop()

## 6) 우선순위큐 (리스트에서 최소원소 추출하는 자료구조), 그리디 알고리즘에서 주로사용
	# heap queue 초기화
 	import heapq
	heap = []

	heapq.heappush(heap, 50)
	heapq.heappush(heap, 10)
	heapq.heappush(heap, 20)
 	heapq.heappop(heap)
	print(heap)
 
# 2.알고리즘

## 1) DFS
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

## 2) BFS
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

## 3) 이진탐색
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

## 4) 소수판별
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
	    

## 5) 순열
	from itertools import permutations
	for i in permutations([1,2,3,4], 2):
    		print(i, end=" ")


## 6)조합 
	from itertools import combinations
	for i in combinations([1,2,3,4], 2):
    		print(i, end=" ")

## 7)동서남북
	import sys
	def func(arr,visited,i,j):
	    #동서남북
	    visited[i][j] = 1
	    answer = 1
	    tmpArr = [[1,0],[-1,0],[0,1],[0,-1]]
	    for arr1 in tmpArr:
	        dx=arr1[0]+i
	        dy=arr1[1]+j
	        if dx >= 0 and dx < n and dy >= 0 and dy < n:
	            if visited[dx][dy] == 0 and arr[dx][dy] == 1:
	                answer += func(arr,visited,dx,dy)
	    return answer
	        
	n = int(input())
	
	arrtot = []
	arr=[]
	cnt=0
	arrtmp=[]
	for _ in range(n):
	    str1 = str(input())
	    for i in str1:
	        arrtot.append(int(i))
	    arr.append(arrtot)
	    arrtot=[]
	visited = [[0] * n for _ in range(n)]
	for i in range(n):
	    for j in range(n):
	        if visited[i][j] == 0 and arr[i][j] == 1:
	            answer = func(arr,visited,i,j)
	            cnt+=1
	            arrtmp.append(answer)
	arrtmp.sort()
	print(cnt)
	for z in arrtmp:
	    print(z)



