import re
import numpy as np
from nltk.corpus import wordnet as wn

words = [re.sub('\t',' ',line).strip('.\r\n') for line in open("Data/NRC.txt")]

#Words by emotion
whappy    = [] #689
wsad      = [] #1191
wdisgust  = [] #1058
wfear     = [] #1476
wsurprise = [] #534
wanger    = [] #1247

for word in words:
    if(word[-1:] == '1'): #we have an emotion
	if(word[-5:-2] == 'joy'):
	    whappy.append(word[:-6])
	elif(word[-9:-2] == 'sadness'):
	    wsad.append(word[:-10])
	elif(word[-10:-2] == 'surprise'):
	    wsurprise.append(word[:-11])
	elif(word[-7:-2] == 'anger'):
	    wanger.append(word[:-8])
	elif(word[-9:-2] == 'disgust'):
	    wdisgust.append(word[:-10])
	elif(word[-6:-2] == 'fear'):
	    wfear.append(word[:-7])

# for comparison with tagged emotions
def EVConverter(EV):
    strings = ['ag', 'dg', 'fr', 'hp', 'ne', 'sd', 'sp']
    indices = [i for i, x in enumerate(EV) if x == max(EV)] # ties are broken by first occurence
    for index in indices:
        return strings[index]

# Returns a list of words related to the input word (IGNORING POS TAGGING), be sure to from nltk.corpus import wordnet as wn
def synsets(word):
    synsets = wn.synsets(word)
    words = []
    for syn in synsets:
        words.append(syn.name().split('.')[0].replace('_',' ').encode('ascii','ignore')) # two word synsets won't be found in database
    words = set(words)
    words = list(words) # remove duplicates
    return words

def kbclassifier(sentences, synMode = 1):
    results = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        sentence = sentence.split(' ')
        EV = [0,0,0,0,0.0,0,0] # ag, dg, fr, hp, ne, sd, sp
        for word in sentence:
            if (whappy.count(word) > 0):
                EV[3] += 1
                EV[4] -= 0.3
            if (wsad.count(word) > 0):
                EV[5] += 1
                EV[4] -= 0.3
            if(wdisgust.count(word) > 0):
                EV[1] += 1
                EV[4] -= 0.3
            if(wfear.count(word) > 0):
                EV[2] += 1
                EV[4] -= 0.3
            if(wsurprise.count(word) > 0):
                EV[6] += 1
                EV[4] -= 0.3
            if(wanger.count(word) > 0):
                EV[0] += 1
                EV[4] -= 0.3
            else:
                EV[4] += 0.1
            if synMode == 1:
                synWords = synsets(word)
                for word in synWords: #Similar algorithm as before with some 
                    if (whappy.count(word) > 0):
                        EV[3] += 0.5
                        EV[4] -= 0.1
                    if (wsad.count(word) > 0):
                        EV[5] += 0.5
                        EV[4] -= 0.1
                    if(wdisgust.count(word) > 0):
                        EV[1] += 0.5
                        EV[4] -= 0.1
                    if(wfear.count(word) > 0):
                        EV[2] += 0.5
                        EV[4] -= 0.1
                    if(wsurprise.count(word) > 0):
                        EV[6] += 0.5
                        EV[4] -= 0.1
                    if(wanger.count(word) > 0):
                        EV[0] += 0.5
                        EV[4] -= 0.1
                    else:
                        EV[4] += 0.01
                    
                
        results.append(EV)
    return results

""" example usage
output1 = classifier(sentences) # probability vectors
output2 = output1
for i in range(len(output2)):
    output2[i] = EVConverter(output2[i])
output2 = np.asarray(output2) # emotion predictions

print(np.mean(output2 == emotions))
print(metrics.classification_report(emotions,output2))
    

"""

