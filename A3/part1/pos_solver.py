import random
import math
import re
import numpy as np
# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
import numpy as np
from collections import Counter
class Solver:
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
        if model == "Simple":
            cost = 0
            for i in range(len(sentence)):
                if sentence[i] in self.emission[label[i]]:
                    cost += math.log(self.emission[label[i]][sentence[i]]) + math.log(self.pos_prob[label[i]])
                else:
                    cost += math.log(1/float(10**8)) + math.log(self.pos_prob[label[i]])
            # cost = sum([((math.log(self.emission[label[i]][sentence[i]])) + (math.log(self.pos_prob[label[i]])) if sentence[i] in self.emission[label[i]] else (math.log(1/float(10**8))) + (math.log(self.pos_prob[label[i]]))) for i in range(len(sentence))])
            return cost
        elif model == "Complex":
            return -999
        elif model == "HMM":
            return -999
        else:
            print("Unknown algo!")

    # Do the training!
    #
    def train(self, data):
        speech = ['adj', 'adv', 'adp', 'conj', 'det', 'noun', 'num', 'pron', 'prt', 'verb', 'x', '.']
        train_data=[]           # initial prob
        for i in range(len(data)):
            train_data.append(data[i][1])

        train_data=np.concatenate(train_data)

        pos_count = np.array(np.unique(train_data, return_counts=True)).T
        sum=0
        for i in pos_count:
            sum+=int(i[1])
        initial_prob_pos = {}
        for i in pos_count:
            initial_prob_pos[i[0]] = int(i[1])/ sum
        #print(initial_prob_pos)
        ########################################################## initial prob

        word_speech=[]
        for i in data:
            for j in range(len(i[0])):
                #word_speech.append(np.array([[i[0][j], i[1][j]]]))
                word_speech.append(tuple([i[0][j],i[1][j]]))

        word_count = Counter(word_speech)
        #print(word_count)

        '''Calculating Posterior Probability'''
        pos_dict = {pos : {} for pos in speech}

        for w in word_count:
            pos_dict[w[1]].update({w[0] : word_count[w]})


        pos_count_dict = {}
        wordpos_list =()
        for line in data:
            for i in range(len(line[0])):
           
                wordpos_list = tuple([line[0][i],line[1][i]])

        pos_prob = {}
        for pos in pos_dict.keys():
            tt = list(pos_dict[pos].values())
            tt_sum = 0
            for i in tt:
                tt_sum += i
            pos_prob[pos] = float(tt_sum/len(wordpos_list))
        # pos_prob = {pos: float(sum(pos_dict[pos].values()))/len(wordpos_list) for pos in pos_dict.keys()}

        for i in pos_count:
            pos_count_dict[i[0]] = int(i[1])
        #print(pos_count_dict)
        emission_prob = {i : {} for i in speech}
        for i in word_count:
            emission_prob[i[1]].update({i[0] : word_count[i]/pos_count_dict[i[1]]})
        #print(emission_prob)

        pair_list = []
        unique_list = []
        trans_count ={}
        trans_prob ={}
        for pos in speech:
            trans_count[pos] = {}
        for pos in speech:
            trans_prob[pos]= {}

        # trans_count = {pos : {} for pos in speech}
        # trans_prob = {pos : {} for pos in speech}
  
        for line in data:
            for i in range(len(line[1])-1):
                pair_list.append(tuple([line[1][i], line[1][i+1]]))
        pair_list = [tuple([line[1][i],line[1][i+1]]) for line in data for i in range(len(line[1])-1)]

        unique_list = list(set(pair_list))
        
        for element in unique_list:
            trans_count[element[0]].update({element[1] : pair_list.count(element)})
        
        for pos in speech:
            trans_prob[pos] = {pos : (1/float(10**8)) for pos in speech}
            for key, v in trans_count[pos].items():
                tt = list(trans_count[pos].values())
                tt_sum = 0
                for i in tt:
                    tt_sum += i
                trans_prob[pos][key] = v/tt_sum
                # trans_prob[pos].update({key: (v/float(sum(list(trans_count[pos].values()))))})    
                

        print('POS: ', pos_prob)

        self.emission = emission_prob
        self.initial = initial_prob_pos
        self.speech =speech
        self.transition = trans_prob
        self.pos_prob = pos_prob

    def post_first(self, word):
        prob = {}
        # if word not in prob.keys():
        #     for j in self.speech:
        #             if word in self.emission[j]:
        #                 prob[word] =  self.emission[j][word] * self.initial[j]
        #             else:
        #                 prob[word]= (prob[word])[j]=(1/float(10**8)) * self.initial[j]
        if word not in prob.keys():
            for pos in self.speech:
                if word in self.emission[pos]:
                    prob[word] = {pos: self.emission[pos][word] * self.initial[pos]}
                else:
                    prob[word] = {pos: (1/float(10**8)) * self.initial[pos]}
        # if word not in prob.keys():
           
        #     prob[word] = {pos: self.emission[pos][word] * self.initial[pos] if word in self.emission[pos]
        #                   else (1/float(10**8)) * self.initial[pos] for pos in self.speech}
        return prob[word]
    
    def post_second(self, word, prev_pos):
        prob = {}
        if word not in prob.keys():
            for pos in self.speech:
                if word in self.emission[pos]:
                   prob[word]={pos: self.emission[pos][word] * (float(self.transition[prev_pos][pos] * self.pos_prob[prev_pos]) / self.pos_prob[pos]) * self.pos_prob[pos] }
                else:
                    prob[word]={pos:1/float(10**8) * (float(self.transition[prev_pos][pos] * self.pos_prob[prev_pos]) / self.pos_prob[pos]) * self.pos_prob[pos] }
            # prob[word] = {pos : self.emission[pos][word] * (float(self.transition[prev_pos][pos] * self.pos_prob[prev_pos]) / self.pos_prob[pos])* self.pos_prob[pos] if word in self.emission[pos] else (1/float(10**8)) * (float(self.transition[prev_pos][pos] * self.pos_prob[prev_pos]) / self.pos_prob[pos])* self.pos_prob[pos] for pos in self.speech}
            # print("probbb",prob[word])
            
            
        return prob[word]
    
    def post_other(self, word, prev_pos, prev_pos_sec):
        prob = {}
        if word not in prob.keys():
           
            for pos in self.speech:
                if word in self.emission[pos]:
                    prob[word] = {pos: self.emission[pos][word] * (float(self.transition[prev_pos][pos] * self.pos_prob[prev_pos]) / self.pos_prob[pos]) * (float(self.transition[prev_pos_sec][pos] * self.pos_prob[prev_pos_sec]) / self.pos_prob[pos]) * self.pos_prob[pos]}
                else:
                    prob[word]={pos:1/float(10**8) * (float(self.transition[prev_pos][pos] * self.pos_prob[prev_pos]) / self.pos_prob[pos]) * (float(self.transition[prev_pos_sec][pos] * self.pos_prob[prev_pos_sec]) / self.pos_prob[pos]) * self.pos_prob[pos]}
  
            # prob[word] = {pos : self.emission[pos][word] * (float(self.transition[prev_pos][pos] * self.pos_prob[prev_pos]) / self.pos_prob[pos]) * (float(self.transition[prev_pos_sec][pos] * self.pos_prob[prev_pos_sec]) / self.pos_prob[pos] )* self.pos_prob[pos] if word in self.emission[pos] else (1/float(10**8)) * (float(self.transition[prev_pos][pos] * self.pos_prob[prev_pos]) / self.pos_prob[pos]) * (float(self.transition[prev_pos_sec][pos] * self.pos_prob[prev_pos_sec]) / self.pos_prob[pos]) * self.pos_prob[pos] for pos in self.speech}
            
        return prob[word]   
    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        speech_of_word = []
        for i in range(len(sentence)):
            prob = []
            for j in self.speech:
                if sentence[i] in self.emission[j]:
                    prob.append(self.initial[j] + self.emission[j][sentence[i]])
                else:
                    prob.append(self.initial[j] + np.log(.000000001))
            word_index = np.argmax(prob)
            speech_of_word.append(self.speech[word_index])
        return speech_of_word

    def complex_mcmc(self, sentence):
        iteration = 5000
        warmup = 2000
        
        pos_mcmc_dict = {"POS_" + str(i) : {} for i in range(len(sentence))}
        
        sequence = ['noun'] * len(sentence)
        for i in range(len(sentence)):
            if i == 0:
                prob = {}

                if sentence[i] not in prob.keys():
                    for pos in self.speech:
                        if sentence[i] in self.emission[pos]:
                            prob[sentence[i]] = {pos: self.emission[pos][sentence[i]] * self.initial[pos]}
                        else:
                            prob[sentence[i]] = {pos: (1/float(10**8)) * self.initial[pos]}
                        # if sentence[i] not in prob.keys():
                        #     prob[sentence[i]] = {pos: self.emission[pos][sentence[i]] * self.initial[pos] if sentence[i] in self.emission[pos]
                        #         else (1/float(10**8)) * self.initial[pos] for pos in self.speech}

                prob_first= prob[sentence[i]]
                # prob_first = self.post_first(sentence[i])
                 
           
                    
                sample_first = list(np.random.choice([keys for keys in prob_first.keys()], iteration, p = [float(prob_first[keys])/sum(prob_first.values()) for keys in prob_first.keys()]))
                
                sample_first = sample_first[warmup:] 

                pos_mcmc_dict["POS_" + str(i)] = {pos :  (float(sample_first.count(pos))/len(sample_first)) for pos in self.speech }

                sequence[i] = max(pos_mcmc_dict["POS_" + str(i)], key = pos_mcmc_dict["POS_" + str(i)].get)
            
            elif i == 1:
                # prob_second = self.post_second(sentence[i], sequence[i-1]) 
                if sentence[i] not in prob.keys():
                    for pos in self.speech:
                        if sentence[i] in self.emission[pos]:
                            prob[sentence[i]]={pos: self.emission[pos][sentence[i]] * (float(self.transition[sequence[i-1]][pos] * self.pos_prob[sequence[i-1]]) / self.pos_prob[pos]) * self.pos_prob[pos] }
                        else:
                            prob[sentence[i]]={pos:1/float(10**8) * (float(self.transition[sequence[i-1]][pos] * self.pos_prob[sequence[i-1]]) / self.pos_prob[pos]) * self.pos_prob[pos] }

                prob_second= prob[sentence[i]]            
                sample_second = list(np.random.choice([keys for keys in prob_second.keys()], iteration, p = [float(prob_second[keys])/sum(prob_second.values()) for keys in prob_second.keys()]))
                sample_second = sample_second[warmup:] 
                
                pos_mcmc_dict["POS_" + str(i)] = {pos :  (float(sample_second.count(pos))/len(sample_second)) for pos in self.speech }

                sequence[i] = max(pos_mcmc_dict["POS_" + str(i)], key = pos_mcmc_dict["POS_" + str(i)].get)
                
            else:
                # prob_other= self.post_other(sentence[i], sequence[i-1], sequence[i-2]) 
                if sentence[i] not in prob.keys():
           
                    for pos in self.speech:
                        if sentence[i] in self.emission[pos]:
                            prob[sentence[i]] = {pos: self.emission[pos][sentence[i]] * (float(self.transition[sequence[i-1]][pos] * self.pos_prob[sequence[i-1]]) / self.pos_prob[pos]) * (float(self.transition[sequence[i-2]][pos] * self.pos_prob[sequence[i-2]]) / self.pos_prob[pos]) * self.pos_prob[pos]}
                        else:
                            prob[sentence[i]]={pos:1/float(10**8) * (float(self.transition[sequence[i-1]][pos] * self.pos_prob[sequence[i-1]]) / self.pos_prob[pos]) * (float(self.transition[sequence[i-2]][pos] * self.pos_prob[sequence[i-2]]) / self.pos_prob[pos]) * self.pos_prob[pos]}
                prob_other = prob[sentence[i]]
                sample_other = list(np.random.choice([keys for keys in prob_other.keys()], iteration, p = [float(prob_other[keys])/sum(prob_other.values()) for keys in prob_other.keys()]))
                sample_other = sample_other[warmup:] 
                
                pos_mcmc_dict["POS_" + str(i)] = {pos :  (float(sample_other.count(pos))/len(sample_other)) for pos in self.speech }

                sequence[i] = max(pos_mcmc_dict["POS_" + str(i)], key = pos_mcmc_dict["POS_" + str(i)].get)

        print(sequence)
        return sequence

    def hmm_viterbi(self, sentence):
        viterbi_dict ={}
        for i in range(len(sentence)):
            viterbi_dict[i] =  {}
        # viterbi_dict = {i : {} for i in range(len(sentence))}
        
        for i in range(len(sentence)):
            temp_dict = {}
            if i  == 0:
                for j in self.speech:
                    if sentence[i] in self.emission[j]:
                        temp_dict[j] = tuple([self.initial[j] * self.emission[j][sentence[i]], j])
                    else:
                        # print('INIT',self.initial)
                        # print('POS: ', j)
                        temp_dict[j] = tuple([self.initial[j] * float(1/10**8), j]) 
                    viterbi_dict[i] = temp_dict
         
            else:
                 
                for pos in self.speech:
                    if sentence[i] in self.emission[pos]:
                        emi = self.emission[pos][sentence[i]]
                    else:
                        emi =  ((1/float(10**8)))
                    min_val = max([ ((viterbi_dict[i-1][pos_prev][0] * self.transition[pos_prev][pos]), pos_prev) for pos_prev in self.speech])

                    temp_dict[pos] = tuple([emi * min_val[0], min_val[1]])
                    
                viterbi_dict[i] = temp_dict
            i = i + 1

        pos_list = []
        prev = ""

        for i in range(len(sentence) - 1, -1, -1):
            
            if i == len(sentence) - 1:
                minimum = max(list(viterbi_dict[i].values()))

                for each in viterbi_dict[i].keys():
                    if viterbi_dict[i][each][0] == minimum[0] and viterbi_dict[i][each][1] == minimum[1]:

                        pos_list.append(each)
                        prev = minimum[1]
            else:
                pos_list.append(prev)
                prev = viterbi_dict[i][prev][1]
                    
                
                    
        pos_list.reverse()
       
        
        return pos_list
        



    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")