def fetchTweets():
	f = open('KashmirTwitterData.txt', 'r')
	l = []

	for line in f.readlines():
		try:
			_id = line.split('\t')[3].strip()
			twt = line.split('\t')[4].strip()
			l.append({_id: twt})

		except IndexError:
			continue

	f.close()

	return l

def main():
	l = fetchTweets()
	for x in l :
		print(x)
		print('--------------------------')


if __name__ == '__main__':
	main()