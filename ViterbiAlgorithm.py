import numpy as np

def Algorithm_Of_Viterbi(state_transition, initial_state, Emissions, Given_sequence):
    
    I = state_transition.shape[0]    #get number of states
    N = Given_sequence.shape[1]  #get length of sequence
    
    # Compute log probabilities with tiny
    tiny = np.finfo(0.).tiny
    state_transition_log = np.log(state_transition + tiny)
    initial_state_log = np.log(initial_state + tiny)
    Emissions_log = np.log(Emissions + tiny)

    
    # initialize D and state_path matrices
    probability_scores = np.zeros([I, N])
    state_path = np.zeros([I, N-1])

    probability_scores[:, 0] = initial_state_log + Emissions[:, 0]  # find probabilities for each state for the first observation character

    # compute probability_scores and state_path in a nested loop
    for n in range(1, N): #for each sequence symbol, we start from 1 because the previous probabilities were found
        for i in range(I): #for each state
	
	    # add each probability of the states with the n-1 'probability_scores' probabilities we previously found
            tempor = state_transition_log[:, i] + probability_scores[:, n-1] # n starts with 1 so we want n-1


	    # then get the max (array max) probability and add it by the Emission for the sequence symbol
	    # we want to produce for each state then we store our new probabilities to array D
   
            probability_scores[i,n] = np.amax(tempor) + Emissions_log[i, Given_sequence[0, n]-1] 
           
	    # get the index of the max value which means get the state that has the max probability
	    state_path[i, n-1] = np.argmax(tempor)
            

    Best_path = np.zeros([1, N])


    # get the best last state from probability_scores array as we don't save it into state_path array
    # we insert it at the end of the Best_path array

    Best_path[0, -1] = np.argmax(probability_scores[:, -1])
    

    # Backtracking process begins
    for n in range(N-2, 0, -1): # N-2 : array indexes are 1 less than length of N and also we have already found last step of path previously

	# find optimal state
        Best_path[0, n] = state_path[int(Best_path[0, n+1]), n]
    

    # Convert zero-based indices to state indices
    Optimal_state_sequence = Best_path.astype(int)+1
    

    return Optimal_state_sequence


def convert_SeqToNumbers(seq):

	#convert sequence characters to numbers
	for r in (("A", "1"), ("C", "2"), ("G", "3"), ("T", "4")):
		seq = seq.replace(*r)

	#convert string to list
	seq = list(seq)

	#convert string list to int list
	for i in range(0, len(seq)): 
		seq[i] = int(seq[i]) 


	return seq


def convert_NumbersToState(numbers):

	#convert int list to string list
	numbers = map(str, numbers)  
	
	#join list items into a single string
	numbers = ''.join(numbers)     

	#convert numbers to states
	for r in (("1", "a"), ("2", "b")):
		numbers = numbers.replace(*r)

	return numbers



#state transition probabilities
state_transition = np.array([[0.9, 0.1], 
             		    [0.1, 0.9]])


#initial state probabilities
initial_state  = np.array([[0.5, 0.5]])


#emissions probabilities
emissions  = np.array([[0.4, 0.4, 0.1, 0.1], 
              		[0.3, 0.3, 0.2, 0.2]])


#the sequence we are going to call viterbi with
given_seq = "GGCT"

#convert the sequence to numbers
seqAsNumbers = convert_SeqToNumbers(given_seq)

#create array based on the numbers
seqAsNumbersArray = np.array([seqAsNumbers])


# Apply Viterbi algorithm
Best_States_As_Numbers = Algorithm_Of_Viterbi(state_transition, initial_state, emissions, seqAsNumbersArray)

#convert optimal path from numbers to symbols
Best_states = convert_NumbersToState(Best_States_As_Numbers)


print('Given sequence as symbols:   '+given_seq)
print('Given sequence as numbers:   '+str(seqAsNumbers))
print('Optimal state sequence: '+Best_states)





