# The purpose of this file is to parse all of the annotated sentences into 6 simple list objects
# 2 for each data source, one with the sentences and the other with the labels.
# There is also a preparse function which replaces some punctuation and emoticons with words

import re

# rewrite rules based on brooks top features
def preparse(sentence):
  newSentence = sentence.replace(':)',' HAPPYFACE ')
  newSentence = newSentence.replace(':-)',' HAPPYFACE ')
  newSentence = newSentence.replace(';)',' WINKYFACE ')
  newSentence = newSentence.replace(';-)',' WINKYFACE ')
  newSentence = sentence.replace(':(',' SADFACE ')
  newSentence = newSentence.replace(':-(',' SADFACE ')
  newSentence = sentence.replace(':o',' SURPRISEFACE ')
  newSentence = newSentence.replace(':O',' SURPRISEFACE ')
  # ...
  newSentence = re.sub(r'\.{2,}',' ELLIPIS ',newSentence)
  # ?
  newSentence = re.sub(r'\?{2,}',' QUESTIONMARKS ', newSentence)
  newSentence = re.sub(r'\?',' QUESTIONMARK ', newSentence)
  # !
  newSentence = re.sub(r'\!{2,}',' EXCLAMATIONMARKS ', newSentence)
  newSentence = re.sub(r'\!',' EXCLAMATIONMARK ', newSentence)
  # cleanup whitespace
  newSentence = re.sub(r'\s{2,}',' ',newSentence)
  newSentence = re.sub(r'\s$','',newSentence)
  
  return newSentence

def preparseDeep(sentence):
    # Potential work similar to Mike Thelwal's
    return sentence

# AMAN 2007, 4090 'gold' blog sentences using hp sd fr ag sp dg ne (no mixed emotion me) (7 usable emotions)

#import each line as an index (regex strips numbers, the strip removes newlines and the lone spaces)
amanSentences = [re.sub('\d+.\s','',line).strip('.\r\n') for line in open("Data/Amangold.txt")]
amanEmotions = []

#alphanumeric and spaces + lowercase + split into individual words
for i in range(len(amanSentences)):
    amanSentences[i] = preparse(amanSentences[i])
    amanSentences[i] = re.sub(r'[^a-zA-Z0-9\s]','',amanSentences[i]).lower().split(' ')
    amanEmotions.append(amanSentences[i][0])
    amanSentences[i].remove(amanEmotions[i])
    
# ISEAR, 7666 joy fear anger disgust guilt shame sadness (5 emotions usable)

isear = [line.strip('.\r\n') for line in open("Data/isear.txt")] # grab sentences and remove periods, newline chars
isearEmotions = []
isearSentences = []

for i in range(len(isear)):
  isear[i] = re.sub(',',' ',isear[i])
  isear[i] = preparse(isear[i])
  isear[i] = re.sub(r'[^a-zA-Z0-9\s]','',isear[i]).lower().split(' ')
  while isear[i].count(''):
     isear[i].remove('')
  isearSentences.append(isear[i][3:])
  isearEmotions.append(isear[i].pop(0))
  
  if isearEmotions[i] == 'joy':
    isearEmotions[i] = 'hp'
  elif isearEmotions[i] == 'fear':
    isearEmotions[i] = 'fr'
  elif isearEmotions[i] == 'anger':
    isearEmotions[i] = 'ag'
  elif isearEmotions[i] == 'disgust':
    isearEmotions[i] = 'dg'
  elif isearEmotions[i] == 'sadness':
    isearEmotions[i] = 'sd'
  
# remove shame and guilt entirely

while isearEmotions.count('shame'):
  index = isearEmotions.index('shame')
  del isearEmotions[index]
  del isearSentences[index]

while isearEmotions.count('guilt'):
  index = isearEmotions.index('guilt')
  del isearEmotions[index]
  del isearSentences[index]

  
# SemEval 2007, 1250 headlines fine grained into anger disgust fear joy sadness surprise

semevalSentences = [line.strip('.\r\n') for line in open("Data/semeval.txt")]
semevalEmotions = [line.strip('.\r\n') for line in open("Data/semeval2.txt")]
emotionClasses = ['ag','dg','fr','hp','sd','sp','ne']

#semeval emotions must be converted into coarse categories.
for i in range(len(semevalEmotions)):
  semevalEmotions[i] = semevalEmotions[i].split(' ')
  semevalEmotions[i].pop(0)
  semevalEmotions[i] = [int(j) for j in semevalEmotions[i]] #convert to type int so that max() works
  if max(semevalEmotions[i]) < 25:
    number = 6
  else:
    number = semevalEmotions[i].index(max(semevalEmotions[i]))
    #if max(semevalEmotions[i]) < 20:
      #print str(max(semevalEmotions[i])) + " with index "+ str(i)
  semevalEmotions[i] = emotionClasses[number]
  
  semevalSentences[i] = preparse(semevalSentences[i])
  semevalSentences[i] = re.sub(r'[^a-zA-Z0-9\s]','',semevalSentences[i]).lower().split(' ')
  

# final output of this script:
# 5477 ISEAR (does not contain sp and ne)
# 4090 AMAN
# 1250 SEMEVAL

# 	ISEAR 	AMAN 	SEMEVAL*
# fr	1095	115	172
# ag	1096	179	68
# dg	1096	172	33
# hp	1094	536	366
# sd	1096	173	240
# sp	0	115	158
# ne	0	2800	211

