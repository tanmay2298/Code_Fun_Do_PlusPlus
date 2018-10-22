import string 

def load (name) : 
	file = open(name, 'r')
	data = file.read() 
	file.close() 
	return data 


def save(descriptions, filename):
	lines = list()
	for key, desc in descriptions.items():
			lines.append(key + ' ' + desc)
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

def make_map(document) : 
	mapping = {} 
	for line in document.split('\n') : 
		words = line.split() 
		if len(line) < 2 : 
			continue
		img_id , img_desc = words[0] , words[1:]
		img_id = img_id.split('.')[0] #jpeg removed 
		img_desc = ' '.join(img_desc) # words combined to give one sentence
		mapping[img_id] = img_desc
	return mapping 
    
def clean(descriptions):
	table = str.maketrans('', '', string.punctuation) # table removes punctuation
	for key, desc in descriptions.items():
			desc = desc.split()
			desc = [word.lower() for word in desc]
			desc = [w.translate(table) for w in desc]
			desc = [word for word in desc if len(word)>1 and word.isalpha() ]


data = load ("tokens.txt")
img_desc = make_map(data)	
clean(img_desc) 
save(img_desc, 'descriptions.txt')