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

## Input
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
   

## 사칙연산
	# 나누기 /
 	print(a / b) 

  	# 나머지 %
  	print(a % b) 

   	# 몫 //
   	print(a // b)

	# 거듭제곱 **
	print(a ** b) 

## 리스트, 배열
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

 	# dict key정렬
  	sorted(dict1.items(), key=lambda x: x[0])
   	
 	# dict value정렬
  	sorted(dict1.items(), key=lambda x: x[1])

## queue (선입선출)
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

## 우선순위큐 (리스트에서 최소원소 추출하는 자료구조), 그리디 알고리즘에서 주로사용
	# heap queue 초기화
 	import heapq
	heap = []

	heapq.heappush(heap, 50)
	heapq.heappush(heap, 10)
	heapq.heappush(heap, 20)
 	heapq.heappop(heap)
	print(heap)
 
# 2.문제풀이

## 동서남북 (DFS=for+재귀, BFS=while+Deque)
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



