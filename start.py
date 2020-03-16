from q_learning import route

optiomal_route = route('E', 'G')
print(optiomal_route)
# ['E', 'I', 'J', 'F', 'B', 'C', 'G']

optiomal_route = route('C', 'L')
print(optiomal_route)
# ['C', 'G', 'H', 'L']

