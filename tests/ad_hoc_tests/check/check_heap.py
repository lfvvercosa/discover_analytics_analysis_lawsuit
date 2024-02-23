import heapq


test = [(1,2,3)]
heapq.heapify(test)

heapq.heappush(test, (1,2,1))
heapq.heappush(test, (1,1,4))


retrieve = heapq.heappop(test)

print(retrieve)