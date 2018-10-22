def load_doc(filename):
	file = open(filename, 'r')
	text = file.read()
	file.close()
	return text

def to_lines(descriptions): # converts dictionary contents to a list 
	all_desc = list()
	for key in descriptions.keys():
		all_desc.append(descriptions[key])
	return all_desc

def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer


def get_max(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)

def train_test(descriptions , features) : 
  train_d , test_d , train_f , test_f = dict() , dict() , dict() , dict()  
  for i,key in enumerate(descriptions) : 
    if ( i < 0.8 * len(descriptions)) : 
      train_d[key] = descriptions[key]
      train_f[key] = features[key]
    else : 
      test_d[key] = descriptions[key]
      test_f[key] = features[key]
  return train_d, test_d , train_f , test_f
 

def load_clean_descriptions(filename):
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		tokens = line.split()
		image_id, image_desc = tokens[0], tokens[1:]
		desc = 'startseq ' + ' '.join(image_desc) + ' endseq' # for show and tell model 
		descriptions[image_id] = desc
	return descriptions

