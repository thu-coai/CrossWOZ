import argparse
import sys
import time

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


#def delexicalise(sent,dact): # for domain4
#	feat = SoftDActFormatter().parse(dact,keepValues=True)
#	return ExactMatchDataLexicaliser().delexicalise(sent,feat['s2v'])
#
#
#def lexicalise(sent,dact): # for domain4
#	feat = SoftDActFormatter().parse(dact,keepValues=True)
#	return ExactMatchDataLexicaliser().lexicalise(sent,feat['s2v'])
#
#
#def parse_sr(sr, domain): # for domain4
#	'''
#	input da: 'inform(name=piperade;goodformeal=dinner;food=basque)'
#	return  : a str 'domain|da|slot1, slot2, ...'
#	Note: cannot deal with repeat slots, e.g. slot_name*2 will has the same sr as slot_name*1
#	'''
#	da = sr.split('(')[0]
#	_sr = sr.split('(')[1].split(')')[0].split(';')
#	slots = []
#	for sv in _sr:
#		slots.append(sv.split('=')[0]) 
#	slots = sorted(slots)
#
#	res = domain + '|' + da + '|'
#	for slot in slots:
#		res  += (slot+',')
#	res = (res[:-1]) # remove last ,
#	return res
#
#
#def score_domain4(res_file):
#	# parse test set to have semantic representation of each target
#	target2sr = {} # target sentence to a defined str of sr
#	sr2content = {}
#	domains = ['restaurant', 'hotel', 'tv', 'laptop']
#	repeat_count = 0
#	for domain in domains:
#		with open('data/domain4/original/'+domain+'/test.json') as f:
#			for i in range(5):
#				f.readline()
#			data = json.load(f)
#	
#		for sr, target, base in data:
#			target = delexicalise( normalize(re.sub(' [\.\?\!]$','',target)),sr)
#			target = lexicalise(target, sr)
#				
#			sr = parse_sr(sr, domain)
#			if target in target2sr:
#				repeat_count += 1
#				continue
#			if target[-1] == ' ':
#				target = target[:-1]
#			target2sr[target] = sr
#	
#			if sr not in sr2content:
#				sr2content[sr] = [[], [], []] # [ [refs], [bases], [gens] ]
#
#	with open(res_file) as f:
#		for line in f:
#			if 'Target' in line:
#				target = line.strip().split(':')[1][1:]
#				sr = target2sr[target]
#				sr2content[sr][0].append(target)
#	
#			if 'Base' in line:
#				base = line.strip().split(':')[1][1:]
#				if base[-1] == ' ':
#					base = base[:-1]
#				sr2content[sr][1].append(base)
#	
#			if 'Gen' in line:
#				gen = line.strip().split(':')[1][1:]
#				sr2content[sr][2].append(gen)
#
#	return sr2content


def score_woz(res_file, ignore=False):
	#corpus = []
	feat2content = {}
	with open(res_file) as f:
		for line in f:
			if 'Feat' in line:
				feat = line.strip().split(':')[1][1:]
	
				if feat not in feat2content:
					feat2content[feat] = [[], [], []] # [ [refs], [bases], [gens] ]
				continue

			if 'Target' in line:
				target = line.strip().split(':')[1][1:]
				if feat in feat2content:
					feat2content[feat][0].append(target)
	
			if 'Base' in line:
				base = line.strip().split(':')[1][1:]
				if base[-1] == ' ':
					base = base[:-1]
				if feat in feat2content:
					feat2content[feat][1].append(base)
	
			if 'Gen' in line:
				gen = line.strip().split(':')[1][1:]
				if feat in feat2content:
					feat2content[feat][2].append(gen)

	return feat2content

def get_bleu(feat2content, template=False, ignore=False):
	test_type = 'base' if template else 'gen'
	print('Start', test_type, file=sys.stderr)

	gen_count = 0
	list_of_references, hypotheses = {'gen': [], 'base': []}, {'gen': [], 'base': []}
	for feat in feat2content:
		refs, bases, gens = feat2content[feat]
		gen_count += len(gens)
		refs = [s.split() for s in refs]
	
		for gen in gens:
			gen = gen.split()
			list_of_references['gen'].append(refs)
			hypotheses['gen'].append(gen)
	
		for base in bases:
			base = base.split()
			list_of_references['base'].append(refs)
			hypotheses['base'].append(base)
	
	
	print('TEST TYPE:', test_type)
	print('Ignore General Acts:', ignore)
	smooth = SmoothingFunction()
	print('Calculating BLEU...', file=sys.stderr)
	print( 'Avg # feat:', len(feat2content) )
	print( 'Avg # gen: {:.2f}'.format(gen_count / len(feat2content)) )
	BLEU = []
	weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.333, 0.333, 0.333, 0), (0.25, 0.25, 0.25, 0.25)]
	for i in range(4):
		if i == 0 or i == 1 or i == 2:
			continue
		t = time.time()
		bleu = corpus_bleu(list_of_references[test_type], hypotheses[test_type], weights=weights[i], smoothing_function=smooth.method1)
		BLEU.append(bleu)
		print('Done BLEU-{}, time:{:.1f}'.format(i+1, time.time()-t))
	print('BLEU 1-4:', BLEU)
	print('BLEU 1-4:', BLEU, file=sys.stderr)
	print('Done', test_type, file=sys.stderr)
	print('-----------------------------------')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train dialogue generator')
	parser.add_argument('--res_file', type=str, help='result file')
	parser.add_argument('--dataset', type=str, default='woz', help='result file')
	parser.add_argument('--template', type=bool, default=False, help='test on template-based words')
	parser.add_argument('--ignore', type=bool, default=False, help='whether to ignore general acts, e.g. bye')
	args = parser.parse_args()
	assert args.dataset == 'woz' or args.dataset == 'domain4'
	if args.dataset == 'woz':
		assert args.template is False
		feat2content = score_woz(args.res_file, ignore=args.ignore)
	else: # domain4
		assert args.ignore is False
		feat2content = score_domain4(args.res_file)
	get_bleu(feat2content, template=args.template, ignore=args.ignore)
