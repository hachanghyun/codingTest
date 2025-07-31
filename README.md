# 파이썬 공식사이트
## https://docs.python.org/ko/3/

## Library reference
### dict, collections, deque, Counter, defaultdict, heapq, bisect, itertools

# 1.자료구조

## Input
	# 한 줄 입력
	n = input() #문자열
	n = int(input()) #정수

	# 공백 구분 정수를 리스트로 입력 (한줄 입력 받을때)
	arr = list(map(int, input().split()))
	
	# N줄 반복해 2차원 정수 리스트 생성 (N줄 입력 받을때)
	arr2 = [list(map(int, input().split())) for _ in range(N)]
   

## 사칙연산
	# 나누기 / #나누기는 작대기
	# 나머지 % #나머지는 % 퍼센트
	# 몫 // # 몫은 작대기 두개
	# 거듭제곱 **

## 리스트, 배열
	# 리스트 초기화
	arr=[]
	arr=[] * N

	# 2차원 리스트 초기화 (bfs, dfs)
	visited = [[False for _ in range(m)] for _ in range(n)] n: 행개수, m: 열개수 #열이 안쪽에 들어가고 행이 그 열 덩어리들 하나하나 관리하는 개념으로 암기
	
	# 리스트 원소 추가
	arr.append(n)
	
	# 리스트 정렬
	a.sort()
	a.sort(reverse=True)

## dict (key-value)
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
	
	# dict 정렬
	sorted(dict1.items(), key=lambda x: x[0]) #key정렬
	sorted(dict1.items(), key=lambda x: x[1]) #value정렬

## queue (선입선출)
	# queue 초기화
	from collections import deque
	d = deque()
	
	# 데이터 삽입 #기본이 오른쪽삽입이고 왼쪽 삽입은 left 붙이기
	d.appendleft(0)
	d.append(6) 
	
	# 왼쪽 데이터 지우기 #기본이 오른쪽삭제이고 왼쪽 삭제는 left 붙이기
	d.popleft()
	d.pop()

## 우선순위큐 (리스트에서 최소원소 추출하는 자료구조), 그리디 알고리즘에서 주로사용
	# heap queue 초기화
	import heapq
	heap1 = []
	
	heapq.heappush(heap1, 50)
	heapq.heappush(heap1, 10)
	heapq.heappush(heap1, 20)
	heapq.heappop(heap1)
	print(heap1)
 
# 2.문제풀이

## 동서남북,미로찾기 (DFS=for+재귀, BFS=while+deque)
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

## 누적합 찾기(11659)
	import sys
	
	n, m = map(int,input().split())
	
	#print(n, m)
	
	list1 = list(map(int,input().split()))
	
	#print(list1)
	
	list2 = [list(map(int,input().split())) for _ in range(m)]
	
	#print(list2)
	
	sum1 = 0
	sum2 = 0
	#sum3 = 0
	arr1 = []
	
	arr1.append(0)
	for list1detail in list1:
	    sum1 += list1detail
	    arr1.append(sum1)
	#print('arr1',arr1)
	
	for list2detail in list2:
	    #list2detail[0], list2detail[1]
	
	    '''
	    print('list2detail[0]',list2detail[0])
	    print('list2detail[1]',list2detail[1])
	
	    print('arr1[list2detail[0]]',arr1[list2detail[0]-1])
	    print('arr1[list2detail[1]]',arr1[list2detail[1]])
	    '''
	    
	    sum2 = arr1[list2detail[1]] - arr1[list2detail[0]-1]
	    print(sum2)

## 좋은수 찾기(1253) - for while FW
	import sys
	
	n = int(input())
	
	#print('n',n)
	
	arr = list(map(int, input().split()))
	arr.sort()
	
	#print('arr', arr)
	cnt = 0
	
	for i in range(len(arr)):
	
	    goal = arr[i]
	
	    start = 0
	
	    end = len(arr) - 1
	
	    while start < end:
	        if arr[start] + arr[end] == goal:
	            if start == i:
	                start += 1
	
	            elif  end == i:
	                end -= 1
	
	            else:
	                cnt += 1
	                break
	            
	        elif arr[start] + arr[end] > goal:
	            end -= 1
	
	        elif arr[start] + arr[end] < goal:
	            start += 1
	
	print(cnt)

## 수 정렬하기(2750)
	import sys

	n = int(input())
	
	arr = []
	
	for _ in range(n):
	    arr.append(int(input()))
	
	arr.sort()
	
	#print('arr',arr)
	
	for i in arr:
	    print(i)


## DFS (11724)
	import sys
	sys.setrecursionlimit(10**6)
	input = sys.stdin.readline
	
	# dfs 함수
	def dfs(graph, v, visited):
	    visited[v] = True
	    for i in graph[v]:
	        if not visited[i]:
	            dfs(graph, i, visited)
	
	n, m = map(int, input().split()) # 정점의 개수, 간선의 개수
	graph = [[] for _ in range(n+1)]
	for i in range(m):
	    u, v = map(int, input().split())
	    graph[u].append(v)
	    graph[v].append(u)
	
	count = 0 # 연결 노드의 수
	visited = [False] * (n+1)
	for i in range(1, n+1):
	    if not visited[i]:
	        dfs(graph, i, visited)
	        count += 1 # dfs 한 번 끝날 때마다 count+1
	
	print(count)



